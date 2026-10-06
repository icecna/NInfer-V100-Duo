// CPU-only unit tests for HostPrefixTable: membership, id assignment, LRU stamping, the in-flight
// claim guard, and evict callback ordering. Matching (PrefixIndex) and retention policy live in
// their own units and are tested separately. The payload is a fake (no CUDA, no arena).

#include "targets/qwen3_6/impl/runtime/host_prefix_table.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace ninfer::targets::qwen3_6::detail;

int failures = 0;

int fail(const std::string& message) {
    std::printf("FAIL: %s\n", message.c_str());
    return 1;
}

int expect(bool condition, const std::string& label) {
    return condition ? 0 : fail(label);
}

struct FakeIdentity {
    std::int32_t marker = 0;
    std::vector<std::int32_t> tokens;
};

struct FakePayload {
    FakeIdentity identity;
    std::size_t reuse_boundary = 0;
    std::size_t occupied_bytes = 0;

    std::size_t reuse_boundary_bytes() const noexcept { return reuse_boundary; }
};

namespace {

int test_insert_and_find() {
    HostPrefixTable<FakePayload> table;
    HostPrefixId id = 0;
    const HostPrefixId a = table.insert(FakePayload{.occupied_bytes = 10}, [&] { return id++; });
    const HostPrefixId b = table.insert(FakePayload{.occupied_bytes = 20}, [&] { return id++; });
    if (a != 0 || b != 1) { return fail("id provider sequence"); }
    if (!expect(table.size() == 2, "size after 2 inserts")) { return failures; }
    const auto* ea = table.find(a);
    const auto* eb = table.find(b);
    if (!expect(ea != nullptr && ea->payload.occupied_bytes == 10, "find a")) { return failures; }
    if (!expect(eb != nullptr && eb->payload.occupied_bytes == 20, "find b")) { return failures; }
    if (table.find(99) != nullptr) { return fail("find unknown id"); }
    // Stamping by id only touches that entry.
    table.stamp(b, 7);
    if (!expect(table.find(b)->lru_stamp == 7 && table.find(a)->lru_stamp == 0,
                "stamp affects only the named entry")) {
        return failures;
    }
    table.stamp(99, 3); // unknown id is a no-op
    return 0;
}

int test_in_flight_claim_guard() {
    HostPrefixTable<FakePayload> table;
    HostPrefixId id = 0;
    const HostPrefixId a = table.insert(FakePayload{}, [&] { return id++; });
    table.begin_claim(a);
    if (!expect(table.in_flight_claims() == 1, "claim counted")) { return failures; }
    // Nested claims need nested completion.
    table.begin_claim(a);
    table.end_claim(a);
    if (!expect(table.find(a)->in_flight_claims == 1, "nested claim not fully released")) {
        return failures;
    }
    table.end_claim(a);
    if (!expect(table.in_flight_claims() == 0, "all claims released")) { return failures; }
    // Underflow is a no-op, not a wrap.
    table.end_claim(a);
    if (!expect(table.find(a)->in_flight_claims == 0, "claim underflow wrapped")) {
        return failures;
    }
    // Claims on unknown ids are ignored.
    table.begin_claim(99);
    table.end_claim(99);
    if (!expect(table.in_flight_claims() == 0, "unknown-id claims ignored")) { return failures; }
    return 0;
}

int test_evict_callback_and_removal() {
    HostPrefixTable<FakePayload> table;
    HostPrefixId id = 0;
    const HostPrefixId a = table.insert(FakePayload{}, [&] { return id++; });
    const HostPrefixId b = table.insert(FakePayload{}, [&] { return id++; });
    std::vector<HostPrefixId> evicted;
    if (!expect(table.evict(a, [&](HostPrefixId i) { evicted.push_back(i); }), "evict a")) {
        return failures;
    }
    if (!expect(evicted.size() == 1 && evicted[0] == a, "callback got the id")) {
        return failures;
    }
    if (table.find(a) != nullptr) { return fail("evicted entry still findable"); }
    if (!expect(table.size() == 1 && table.find(b) != nullptr, "survivor intact")) {
        return failures;
    }
    if (table.evict(a)) { return fail("double evict succeeded"); }
    if (table.evict(kInvalidHostPrefixId)) { return fail("evict invalid id succeeded"); }
    table.clear();
    if (!expect(table.empty(), "clear empties the table")) { return failures; }
    return 0;
}

} // namespace

int main() {
    failures += test_insert_and_find();
    failures += test_in_flight_claim_guard();
    failures += test_evict_callback_and_removal();
    if (failures == 0) { std::printf("ok\n"); return 0; }
    std::printf("%d host prefix table check(s) failed\n", failures);
    return 1;
}
