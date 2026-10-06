// Host-only contract test for the KV page ownership registry and its address spaces: reference
// counting, source pins, checkpoint coverage protection, and inactive truncation. No device is
// needed.

#include "targets/qwen3_6/impl/runtime/kv_page_registry.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using ninfer::targets::qwen3_6::detail::KvAddressSpace;
using ninfer::targets::qwen3_6::detail::KvPageRegistry;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int test_page_lifetime() {
    int failures = 0;
    KvPageRegistry registry(4);
    failures += check(registry.capacity() == 4 && registry.occupied() == 0,
                      "registry starts empty");

    const auto page = registry.materialize();
    failures += check(page.has_value(), "materialize returned a page");
    failures += check(registry.occupied() == 1 && registry.references(*page) == 1 &&
                          registry.writer_references(*page) == 0,
                      "materialize grants one reference and no writer");
    failures += check(registry.can_pin_source(*page), "an unwritten page can be pinned");
    registry.set_writer(*page);
    failures += check(!registry.can_pin_source(*page),
                      "a page with an active writer cannot be pinned");
    failures += check(!registry.can_release_reference(*page),
                      "a written page cannot be released before its writer");

    registry.release_writer(*page);
    failures += check(registry.can_pin_source(*page), "released writer allows a source pin");
    registry.pin_source(*page);
    failures += check(!registry.can_release_reference(*page), "a source pin blocks release");
    registry.unpin_source(*page);
    failures += check(registry.can_release_reference(*page), "unpinned page can be released");

    registry.release_reference(*page);
    failures += check(registry.occupied() == 0, "last reference returns the page to the free list");

    // A host replica keeps the page descriptor alive after its last reference.
    const auto resident = registry.materialize();
    registry.attach_host_replica(*resident);
    registry.release_reference(*resident);
    failures += check(registry.occupied() == 1 && registry.host_resident(*resident),
                      "host replica outlives the last reference");
    registry.detach_host_replica(*resident);
    failures += check(registry.occupied() == 0, "detaching the host replica recycles the page");
    return failures;
}

int test_shared_pages() {
    int failures = 0;
    KvPageRegistry registry(8);
    KvAddressSpace owner = KvAddressSpace::create(registry, 3);
    failures += check(owner.pages().size() == 3, "address space materialized three pages");
    for (const std::uint32_t page : owner.pages()) {
        failures += check(registry.references(page) == 1, "creator holds one reference");
    }

    {
        KvAddressSpace shared = KvAddressSpace::fork(registry, owner);
        for (const std::uint32_t page : shared.pages()) {
            failures += check(registry.references(page) == 2, "fork retained a reference");
        }
    }
    for (const std::uint32_t page : owner.pages()) {
        failures += check(registry.references(page) == 1,
                          "releasing the fork dropped only its own reference");
    }
    failures += check(registry.occupied() == 3, "shared pages stay materialized");
    return failures;
}

int test_checkpoint_protection_and_truncate() {
    int failures = 0;
    KvPageRegistry registry(8);
    KvAddressSpace space = KvAddressSpace::create(registry, 4, /*tokens_per_page=*/64);
    space.commit_frontier(256);
    for (const std::uint32_t page : space.pages()) {
        failures += check(registry.committed_columns(page) == 64, "commit_frontier covers pages");
    }

    space.set_checkpoint_requirement(128);
    failures += check(registry.protected_columns(space.pages()[0]) == 64 &&
                          registry.protected_columns(space.pages()[1]) == 64 &&
                          registry.protected_columns(space.pages()[2]) == 0,
                      "checkpoint protection covers only the fully covered prefix");
    failures += check(space.can_truncate_inactive_prefix(128),
                      "unprotected suffix is truncatable");
    space.truncate_inactive_prefix(128);
    failures += check(space.pages().size() == 2 && registry.occupied() == 2,
                      "truncation released the suffix pages");
    failures += check(!space.can_truncate_inactive_prefix(0),
                      "protected pages block truncation below the checkpoint frontier");

    // An active reference also blocks truncation.
    KvPageRegistry other_registry(4);
    KvAddressSpace active = KvAddressSpace::create(other_registry, 2, 64);
    active.commit_frontier(128);
    const std::uint32_t tail = active.pages()[1];
    other_registry.retain_active_reference(tail);
    failures += check(!active.can_truncate_inactive_prefix(64),
                      "an active reference blocks truncation");
    other_registry.release_active_reference(tail);
    failures += check(active.can_truncate_inactive_prefix(64),
                      "dropping the active reference re-allows truncation");
    return failures;
}

int test_prefix_view() {
    int failures = 0;
    KvPageRegistry registry(8);
    KvAddressSpace source = KvAddressSpace::create(registry, 4, /*tokens_per_page=*/64);
    source.commit_frontier(256);

    // A frontier inside a page: the fully covered pages are shared and the partial tail is owned.
    KvAddressSpace view = KvAddressSpace::fork_prefix(registry, source, 200);
    failures += check(view.pages().size() == 4, "prefix view page count");
    for (std::size_t index = 0; index < 3; ++index) {
        failures +=
            check(view.pages()[index] == source.pages()[index], "whole page shared by identity");
        failures += check(registry.references(view.pages()[index]) == 2,
                          "shared page gained a reference");
    }
    failures += check(view.pages()[3] != source.pages()[3], "tail page is owned, not shared");
    failures += check(registry.committed_columns(view.pages()[3]) == 200 % 64,
                      "tail page commits only the frontier remainder");
    failures += check(view.committed_frontier() == 200, "prefix view frontier");

    // A page-aligned frontier needs no tail page.
    KvAddressSpace aligned = KvAddressSpace::fork_prefix(registry, source, 192);
    failures += check(aligned.pages().size() == 3, "aligned prefix view owns no tail");
    aligned.release();

    // Releasing the source must not free pages the view still references.
    source.release();
    for (std::size_t index = 0; index < 3; ++index) {
        failures += check(registry.references(view.pages()[index]) == 1,
                          "shared page survives the source release");
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_page_lifetime();
    failures += test_shared_pages();
    failures += test_checkpoint_protection_and_truncate();
    failures += test_prefix_view();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
