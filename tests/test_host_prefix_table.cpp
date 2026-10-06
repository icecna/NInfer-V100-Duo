// CPU-only unit tests for HostPrefixTable (P1): membership, content lookup with deepest-match
// selection, LRU ordering, the in-flight claim guard, and evict callback ordering.
// The payload is a fake (no CUDA, no arena): token vector + identity stand-in + boundary +
// a bytes counter standing in for the arena extents' occupied bytes.

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

struct FakePrompt {
    std::vector<std::int32_t> tokens;
};

// Fake identity: a marker token plus the resident token run (mirrors prefix_matches' shape:
// token prefix + identity check).
struct FakeIdentity {
    std::int32_t marker = 0;
    std::vector<std::int32_t> tokens;
};

struct FakePayload {
    using Prompt = FakePrompt;
    FakeIdentity identity;
    std::size_t reuse_boundary = 0;
    std::size_t occupied_bytes = 0;

    std::size_t reuse_boundary_bytes() const noexcept { return reuse_boundary; }
};

// Match depth: the resident identity run the prompt reproduces (0 = no match). Mirrors the
// production match_entry: the full identity prefix must match, and the returned depth is the
// usable boundary.
std::size_t match_depth(const FakePrompt& prompt, const FakePayload& payload) {
    if (prompt.tokens.empty() || prompt.tokens[0] != payload.identity.marker) { return 0; }
    const std::size_t count = payload.identity.tokens.size();
    if (count > prompt.tokens.size()) { return 0; }
    for (std::size_t i = 0; i < count; ++i) {
        if (prompt.tokens[i] != payload.identity.tokens[i]) { return 0; }
    }
    return count;
}

struct Counter {
    std::uint64_t value = 0;
    std::uint64_t operator()() { return ++value; }
};

namespace {

int test_insert_and_find() {
    HostPrefixTable<FakePayload> table;
    HostPrefixId id = 0;
    const HostPrefixId a =
        table.insert(FakePayload{.occupied_bytes = 10}, [&] { return id++; });
    const HostPrefixId b =
        table.insert(FakePayload{.occupied_bytes = 20}, [&] { return id++; });
    if (a != 0 || b != 1) { return fail("id provider sequence"); }
    if (!expect(table.size() == 2, "size after 2 inserts")) { return failures; }
    const auto* ea = table.find(a);
    const auto* eb = table.find(b);
    if (!expect(ea != nullptr && ea->payload.occupied_bytes == 10, "find a")) { return failures; }
    if (!expect(eb != nullptr && eb->payload.occupied_bytes == 20, "find b")) { return failures; }
    if (table.find(99) != nullptr) { return fail("find unknown id"); }
    return 0;
}

int test_find_best_deepest_and_lru_refresh() {
    HostPrefixTable<FakePayload> table;
    HostPrefixId id = 0;
    const std::vector<std::int32_t> run{1, 2, 3, 4};
    // Same marker, different resident-run depths: a 2-token and a 4-token prefix.
    const HostPrefixId shallow = table.insert(
        FakePayload{
            .identity  = FakeIdentity{.marker = 1, .tokens = {1, 2}},
            .reuse_boundary = 2,
            .occupied_bytes = 10},
        [&] { return id++; });
    const HostPrefixId deep = table.insert(
        FakePayload{
            .identity  = FakeIdentity{.marker = 1, .tokens = run},
            .reuse_boundary = 4,
            .occupied_bytes = 20},
        [&] { return id++; });
    Counter counter;
    const FakePrompt prompt{.tokens = run};
    HostPrefixEntry<FakePayload>* best = table.find_best(prompt, match_depth, counter);
    if (!expect(best != nullptr && best->id == deep, "deepest match wins")) {
        return failures;
    }
    if (!expect(counter.value == 1, "on_use called once")) { return failures; }
    // Only the winner is stamped.
    if (!expect(table.find(deep)->lru_stamp == 1 && table.find(shallow)->lru_stamp == 0,
                "only the winner is stamped")) {
        return failures;
    }
    // A prompt matching only the shallow entry claims it.
    const FakePrompt short_prompt{.tokens = {1, 2, 7, 8}};
    best = table.find_best(short_prompt, match_depth, counter);
    if (!expect(best != nullptr && best->id == shallow, "shallow-only match")) {
        return failures;
    }
    if (!expect(table.find(shallow)->lru_stamp == 2, "shallow stamped second")) {
        return failures;
    }
    // Unrelated prompt: no match, no stamp.
    const FakePrompt other{.tokens = {9, 9}};
    if (table.find_best(other, match_depth, counter) != nullptr) { return fail("unrelated match"); }
    if (!expect(counter.value == 2, "unrelated prompt does not stamp")) { return failures; }
    return 0;
}

int test_lru_victim_order() {
    HostPrefixTable<FakePayload> table;
    HostPrefixId id = 0;
    const HostPrefixId a = table.insert(FakePayload{}, [&] { return id++; });
    const HostPrefixId b = table.insert(FakePayload{}, [&] { return id++; });
    const HostPrefixId c = table.insert(FakePayload{}, [&] { return id++; });
    // All unstamped: the first inserted is the stable victim.
    const auto* v = table.oldest_evictable();
    if (!expect(v != nullptr && v->id == a, "unstamped victim is first inserted")) {
        return failures;
    }
    // Stamp b=1, c=3, a=2: b is oldest stamped.
    table.stamp(a, 2);
    table.stamp(b, 1);
    table.stamp(c, 3);
    v = table.oldest_evictable();
    if (!expect(v != nullptr && v->id == b, "smallest nonzero stamp is the victim")) {
        return failures;
    }
    // Stamping b refreshes it to newest: c is now oldest stamped.
    table.stamp(b, 4);
    v = table.oldest_evictable();
    if (!expect(v != nullptr && v->id == c, "refreshed entry is no longer victim")) {
        return failures;
    }
    // An unstamped entry is a victim only when nothing stamped is evictable: evict the two
    // stamped ones, then the (originally) unstamped one... all three were stamped; instead
    // insert an unstamped d with all others gone.
    table.evict(a);
    table.evict(b);
    table.evict(c);
    const HostPrefixId d = table.insert(FakePayload{}, [&] { return id++; });
    v = table.oldest_evictable();
    if (!expect(v != nullptr && v->id == d, "lone unstamped entry is evictable")) {
        return failures;
    }
    return 0;
}

int test_in_flight_claim_guard() {
    HostPrefixTable<FakePayload> table;
    HostPrefixId id = 0;
    const HostPrefixId a = table.insert(FakePayload{}, [&] { return id++; });
    const HostPrefixId b = table.insert(FakePayload{}, [&] { return id++; });
    table.stamp(a, 1);
    table.stamp(b, 2);
    table.begin_claim(a);
    if (!expect(table.in_flight_claims() == 1, "claim counted")) { return failures; }
    const auto* victim = table.oldest_evictable();
    if (!expect(victim != nullptr && victim->id == b, "claimed entry is unevictable")) {
        return failures;
    }
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
    if (!expect(table.oldest_evictable()->id == a, "oldest after release")) {
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
    return 0;
}

// Eviction pressure: keep total resident bytes under a cap by evicting the oldest stamped
// entry; verify survivor sets and that the retain-skip path fires when everything is claimed.
int test_eviction_pressure() {
    HostPrefixTable<FakePayload> table;
    HostPrefixId id = 0;
    const std::size_t cap = 25;
    auto occupied = [&] {
        std::size_t total = 0;
        for (HostPrefixId probe = 0; probe < id; ++probe) {
            if (const auto* e = table.find(probe)) { total += e->payload.occupied_bytes; }
        }
        return total;
    };
    auto try_insert = [&](std::size_t bytes) {
        while (occupied() + bytes > cap) {
            const auto* victim = table.oldest_evictable();
            if (victim == nullptr) { return; } // everything is claimed: retain-skip
            table.evict(victim->id);
        }
        table.insert(FakePayload{.occupied_bytes = bytes}, [&] { return id++; });
    };
    try_insert(10); // a
    table.stamp(0, 1);
    try_insert(10); // b
    table.stamp(1, 2);
    try_insert(10); // c: 30 > 25, evict a (oldest stamped)
    if (!expect(table.size() == 2 && occupied() == 20, "oldest evicted at cap")) {
        return failures;
    }
    if (table.find(0) != nullptr) { return fail("oldest stamped entry survived"); }
    if (!expect(table.find(1) != nullptr && table.find(2) != nullptr, "newer entries kept")) {
        return failures;
    }
    // Everything claimed: a fresh 10-byte insert must retain-skip, not evict.
    table.begin_claim(1);
    table.begin_claim(2);
    const std::size_t before = table.size();
    try_insert(10);
    if (!expect(table.size() == before, "retain-skip under full claims")) { return failures; }
    table.end_claim(1);
    table.end_claim(2);
    return 0;
}

int test_session_filtered_lookup() {
    HostPrefixTable<FakePayload> table;
    HostPrefixId id = 0;
    const std::vector<std::int32_t> run{1, 2, 3};
    auto make = [&](const char* session) {
        return table.insert(
            FakePayload{.identity = FakeIdentity{.marker = 1, .tokens = run},
                        .reuse_boundary = run.size()},
            [&] { return id++; }, session);
    };
    const HostPrefixId s1 = make("s1");
    const HostPrefixId s2 = make("s2");
    const HostPrefixId anon = make("");
    Counter counter;
    const FakePrompt prompt{.tokens = run};

    auto* best = table.find_best(prompt, match_depth, counter, "s1");
    if (!expect(best != nullptr && best->id == s1, "same-session entry wins the tie")) {
        return failures;
    }
    best = table.find_best(prompt, match_depth, counter, "s3");
    if (!expect(best != nullptr && best->id == anon, "unknown session falls back to anonymous")) {
        return failures;
    }
    best = table.find_best(prompt, match_depth, counter, "");
    if (!expect(best != nullptr && best->id == s1, "anonymous lookup accepts the first inserted")) {
        return failures;
    }
    if (!expect(table.find(s2)->lru_stamp == 0, "a foreign-session entry is never stamped")) {
        return failures;
    }
    return 0;
}

int test_demand_aware_eviction() {
    HostPrefixTable<FakePayload> table;
    HostPrefixId id = 0;
    const HostPrefixId a = table.insert(FakePayload{}, [&] { return id++; });
    const HostPrefixId b = table.insert(FakePayload{}, [&] { return id++; });
    table.stamp(a, 1);
    table.stamp(b, 2);
    if (!expect(table.oldest_evictable()->id == a, "lru victim without demand")) {
        return failures;
    }
    table.mark_demand(a, 5);
    if (!expect(table.oldest_evictable()->id == b, "a demanded entry resists eviction")) {
        return failures;
    }
    const HostPrefixId c = table.insert(FakePayload{}, [&] { return id++; });
    if (!expect(table.oldest_evictable()->id == c, "a never-used entry is the preferred victim")) {
        return failures;
    }
    table.mark_demand(b, 9);
    table.evict(c);
    if (!expect(table.oldest_evictable()->id == a,
                "the least recently demanded entry is the victim")) {
        return failures;
    }
    return 0;
}

} // namespace

int main() {
    failures += test_insert_and_find();
    failures += test_find_best_deepest_and_lru_refresh();
    failures += test_lru_victim_order();
    failures += test_in_flight_claim_guard();
    failures += test_evict_callback_and_removal();
    failures += test_eviction_pressure();
    failures += test_session_filtered_lookup();
    failures += test_demand_aware_eviction();
    if (failures == 0) { std::printf("ok\n"); return 0; }
    std::printf("%d host prefix table check(s) failed\n", failures);
    return 1;
}