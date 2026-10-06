// Host-only contract test for the host KV extent store: reserving pins the page sources,
// publishing attaches the host replica and releases the pins, aborting restores the arena, and an
// extent whose pages are all unreferenced is returned to the arena.

#include "targets/qwen3_6/impl/runtime/host_kv_extent_store.h"

#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::HostKVArena;
using ninfer::HostKVPageLayout;
using ninfer::PagedKVPoolSpec;
using ninfer::PagedKVPlaneOrder;
using ninfer::PagedKVPlaneSpec;
using ninfer::plan_host_kv_page_layout;
using ninfer::targets::qwen3_6::detail::HostKvExtentStore;
using ninfer::targets::qwen3_6::detail::KvAddressSpace;
using ninfer::targets::qwen3_6::detail::KvPageRegistry;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

HostKVPageLayout make_layout() {
    PagedKVPoolSpec spec;
    spec.page_group_count      = 8;
    spec.logical_page_capacity = 8;
    spec.table_rows            = 1;
    spec.plane_order           = PagedKVPlaneOrder::PageMajor;
    spec.planes.push_back(PagedKVPlaneSpec{
        .dtype = ninfer::DType::BF16, .leading_extent = 1, .head_extent = 64, .alignment = 256});
    return plan_host_kv_page_layout(spec);
}

int test_publish_and_release() {
    int failures = 0;
    const HostKVPageLayout layout = make_layout();
    HostKVArena arena(layout.page_stride * 8, std::span<const HostKVPageLayout>(&layout, 1));
    KvPageRegistry registry(8);
    HostKvExtentStore store(arena, registry, 4);
    KvAddressSpace space = KvAddressSpace::create(registry, 2, 64);

    for (const std::uint32_t page : space.pages()) { registry.set_writer(page); }
    failures += check(!store.reserve(layout, space.pages()).has_value(),
                      "a written page cannot be reserved as a host replica source");
    for (const std::uint32_t page : space.pages()) { registry.release_writer(page); }

    std::optional<HostKvExtentStore::Reservation> reservation =
        store.reserve(layout, space.pages());
    failures += check(reservation.has_value(), "reserve succeeded");
    failures += check(arena.occupied_bytes() > 0, "reserve consumed arena bytes");
    for (const std::uint32_t page : space.pages()) {
        failures += check(registry.source_pins(page) == 1, "reserve pinned the source page");
    }

    std::optional<HostKvExtentStore::Capability> capability = store.publish(std::move(*reservation));
    failures += check(capability.has_value(), "publish succeeded");
    failures += check(store.occupied() == 1, "one published extent");
    for (const std::uint32_t page : space.pages()) {
        failures += check(registry.host_resident(page), "publish attached the host replica");
        failures += check(registry.source_pins(page) == 0, "publish released the source pin");
    }

    // The address space dropping its references leaves the host replica, which release_unreferenced
    // then reclaims with the extent.
    space.release();
    store.release_unreferenced();
    failures += check(store.occupied() == 0, "unreferenced extent was released");
    failures += check(arena.occupied_bytes() == 0, "released extent returned its arena bytes");
    failures += check(registry.occupied() == 0, "detached pages were recycled");
    return failures;
}

int test_abort_restores_arena() {
    int failures = 0;
    const HostKVPageLayout layout = make_layout();
    HostKVArena arena(layout.page_stride * 8, std::span<const HostKVPageLayout>(&layout, 1));
    KvPageRegistry registry(8);
    HostKvExtentStore store(arena, registry, 4);
    KvAddressSpace space = KvAddressSpace::create(registry, 3, 64);

    {
        std::optional<HostKvExtentStore::Reservation> reservation =
            store.reserve(layout, space.pages());
        failures += check(reservation.has_value(), "reserve succeeded");
    }
    failures += check(store.occupied() == 0 && arena.occupied_bytes() == 0,
                      "abandoned reservation returned its arena bytes");
    for (const std::uint32_t page : space.pages()) {
        failures += check(registry.source_pins(page) == 0, "abort released the source pin");
        failures += check(!registry.host_resident(page), "abort attached no host replica");
    }

    // A shared page keeps the extent alive until every reference is gone.
    auto reservation = store.reserve(layout, space.pages());
    auto capability  = store.publish(std::move(*reservation));
    (void)capability;
    KvAddressSpace shared = KvAddressSpace::fork(registry, space);
    store.release_unreferenced();
    failures += check(store.occupied() == 1, "shared page kept the host extent");
    shared.release();
    store.release_unreferenced();
    failures += check(store.occupied() == 1, "the owning address space still holds the pages");
    space.release();
    store.release_unreferenced();
    failures += check(store.occupied() == 0, "dropping every reference released the extent");
    return failures;
}

int test_shared_extent() {
    int failures = 0;
    const HostKVPageLayout layout = make_layout();
    HostKVArena arena(layout.page_stride * 8, std::span<const HostKVPageLayout>(&layout, 1));
    KvPageRegistry registry(8);
    HostKvExtentStore store(arena, registry, 4);
    KvAddressSpace space = KvAddressSpace::create(registry, 2, 64);

    auto reservation = store.reserve(layout, space.pages());
    auto capability  = store.publish(std::move(*reservation));
    failures += check(store.occupied() == 1, "one published extent");
    auto shared = store.retain(*capability);
    failures += check(store.occupied() == 1, "retain shares the existing extent");
    store.release(std::move(*capability));
    failures += check(store.occupied() == 1 && arena.occupied_bytes() > 0,
                      "a shared extent outlives its first owner");
    store.release(std::move(shared));
    failures += check(store.occupied() == 0 && arena.occupied_bytes() == 0,
                      "the last owner frees the extent");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_publish_and_release();
    failures += test_abort_restores_arena();
    failures += test_shared_extent();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
