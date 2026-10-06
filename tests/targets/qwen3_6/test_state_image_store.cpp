// Host-only contract test for the checkpoint state-image registry and its pinned host pool: role
// transitions, checkpoint references, transfer pins, and raw host-image allocation.

#include "targets/qwen3_6/impl/runtime/state_image_store.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>

namespace {

using ninfer::DType;
using ninfer::HostKVArena;
using ninfer::LinearAttentionStatePoolSpec;
using ninfer::targets::qwen3_6::detail::HostStatePool;
using ninfer::targets::qwen3_6::detail::StateImageRole;
using ninfer::targets::qwen3_6::detail::StateImageSpec;
using ninfer::targets::qwen3_6::detail::StateImageStore;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

StateImageSpec make_spec() {
    StateImageSpec spec;
    spec.linear = LinearAttentionStatePoolSpec{
        .layers = 2, .conv_channels = 8, .conv_width = 4, .value_heads = 2, .value_head_dim = 8,
        .key_head_dim = 8, .slot_count = 4, .conv_dtype = DType::BF16};
    spec.hidden            = 64;
    spec.dflash_local_bytes = 512;
    return spec;
}

int test_roles_and_references() {
    int failures = 0;
    const StateImageSpec spec = make_spec();
    failures += check(spec.host_bytes() > 0 && spec.host_bytes() % 256 == 0,
                      "state image host bytes are non-empty and aligned");

    StateImageStore store(spec, 2);
    failures += check(store.capacity() == 2 && store.occupied() == 0, "store starts empty");

    const auto slot = store.reserve_destination();
    failures += check(slot.has_value() && store.role(*slot) == StateImageRole::ReservedDestination,
                      "reserve_destination marks a reserved slot");
    store.activate(*slot);
    failures += check(store.role(*slot) == StateImageRole::ActiveMutable, "activate marks active");
    store.freeze(*slot);
    failures += check(store.role(*slot) == StateImageRole::CheckpointImmutable &&
                          store.checkpoint_references(*slot) == 1 && store.checkpoint_count() == 1,
                      "freeze makes an immutable checkpoint with one reference");

    failures += check(!store.can_activate(*slot), "a referenced checkpoint cannot be reactivated");
    store.retain_checkpoint_reference(*slot);
    failures += check(store.checkpoint_references(*slot) == 2, "retain added a reference");
    store.release_checkpoint_reference(*slot);
    failures += check(store.checkpoint_references(*slot) == 1, "release dropped one reference");
    store.release_checkpoint_owner(*slot);
    failures += check(store.role(*slot) == StateImageRole::Free && store.occupied() == 0,
                      "releasing the owner recycles the slot");

    // A pin prevents the owner from being released; a zero-reference pinned checkpoint survives
    // until the last unpin recycles it.
    const auto pinned = store.reserve_destination();
    store.activate(*pinned);
    store.freeze(*pinned);
    store.pin(*pinned);
    failures += check(!store.can_release_checkpoint_owner(*pinned), "a pinned checkpoint is held");
    store.release_checkpoint_reference(*pinned);
    failures += check(store.role(*pinned) == StateImageRole::CheckpointImmutable &&
                          store.checkpoint_references(*pinned) == 0,
                      "a pinned checkpoint survives its last reference");
    store.unpin(*pinned);
    failures += check(store.role(*pinned) == StateImageRole::Free,
                      "unpinning a zero-reference checkpoint recycles it");
    return failures;
}

int test_host_pool() {
    int failures = 0;
    ninfer::PagedKVPoolSpec pool_spec;
    pool_spec.page_group_count      = 4;
    pool_spec.logical_page_capacity = 4;
    pool_spec.table_rows            = 1;
    pool_spec.plane_order           = ninfer::PagedKVPlaneOrder::PageMajor;
    pool_spec.planes.push_back(
        ninfer::PagedKVPlaneSpec{.dtype = DType::BF16, .leading_extent = 1, .head_extent = 1,
                                 .alignment = 256});
    const ninfer::HostKVPageLayout layout = ninfer::plan_host_kv_page_layout(pool_spec);
    HostKVArena arena(2048, std::span<const ninfer::HostKVPageLayout>(&layout, 1));
    HostStatePool pool(arena, 1024, 2);
    failures += check(pool.capacity() == 2 && pool.occupied() == 0, "pool starts empty");

    const auto first = pool.allocate();
    failures += check(first.has_value() && pool.data(*first) != nullptr,
                      "first host image is addressable");
    const auto second = pool.allocate();
    failures += check(second.has_value(), "second host image is allocated");
    failures += check(pool.occupied() == 2 && arena.occupied_bytes() == 2048,
                      "both images consume the arena");
    failures += check(!pool.allocate().has_value(), "pool exhaustion fails cleanly");

    pool.release(*first);
    failures += check(pool.occupied() == 1 && arena.occupied_bytes() == 1024,
                      "releasing an image returns its arena bytes");
    failures += check(pool.allocate().has_value(), "released image is reusable");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_roles_and_references();
    failures += test_host_pool();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
