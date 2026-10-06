// Host-only contract test for the engine-level prefix shortlist index.

#include "runtime/engine/context_cache/prefix_index.h"

#include <array>
#include <cstdint>
#include <iostream>
#include <vector>

namespace {

using ninfer::runtime::PrefixIndex;

struct Key {
    std::array<std::uint64_t, 2> digest{};
    std::uint32_t frontier     = 0;
    std::uint32_t identity_tag = 0;

    [[nodiscard]] friend constexpr bool operator==(const Key&, const Key&) noexcept = default;
};

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

Key at(std::uint32_t frontier, std::uint64_t digest = 7, std::uint32_t tag = 3) {
    return Key{.digest = {digest, digest}, .frontier = frontier, .identity_tag = tag};
}

int test_match_order_and_filtering() {
    int failures = 0;
    PrefixIndex<int, Key> index;
    index.insert(1, at(10));
    index.insert(2, at(20));
    index.insert(3, at(30));

    // The incoming key at each frontier equals the stored key.
    const auto key_at = [](std::uint32_t frontier) { return at(frontier); };
    const std::vector<int> up_to_25 = index.matches(25, key_at);
    failures += check(up_to_25.size() == 2 && up_to_25[0] == 2 && up_to_25[1] == 1,
                      "matches returns candidates deepest first below the limit");
    const std::vector<int> all = index.matches(30, key_at);
    failures += check(all.size() == 3 && all[0] == 3, "the frontier at the limit is included");

    // A different incoming digest excludes exactly that frontier.
    const auto changed = [](std::uint32_t frontier) {
        return frontier == 20 ? at(20, /*digest=*/9) : at(frontier);
    };
    const std::vector<int> filtered = index.matches(30, changed);
    failures += check(filtered.size() == 2 && filtered[0] == 3 && filtered[1] == 1,
                      "a digest mismatch removes one candidate");

    // A different identity tag removes every candidate.
    const auto other_tag = [](std::uint32_t frontier) {
        Key key        = at(frontier);
        key.identity_tag = 4;
        return key;
    };
    failures += check(index.matches(30, other_tag).empty(),
                      "an identity-tag mismatch removes every candidate");
    return failures;
}

int test_insert_update_and_erase() {
    int failures = 0;
    PrefixIndex<int, Key> index;
    index.insert(1, at(10));
    index.insert(1, at(20)); // same handle: replace its key
    failures += check(index.size() == 1 && index.contains(1), "re-insert updates one entry");

    const auto key_at = [](std::uint32_t frontier) { return at(frontier); };
    const std::vector<int> matched = index.matches(25, key_at);
    failures += check(matched.size() == 1 && matched[0] == 1,
                      "the updated frontier is the one that matches");

    index.erase(1);
    failures += check(index.empty() && !index.contains(1), "erase removes the entry");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_match_order_and_filtering();
    failures += test_insert_update_and_erase();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
