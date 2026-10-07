// Host-only contract test for the context-cache retention policy: session lineage, candidate
// visibility, demand-ordered victims, and the gain-above-loss admission rule.

#include "runtime/engine/context_cache/retention.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

using ninfer::runtime::CheckpointRole;
using ninfer::runtime::ContextRetentionPolicy;
using ninfer::runtime::ContinuationOwnerToken;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

bool has(const std::vector<ContextRetentionPolicy::Candidate>& candidates, std::uint64_t id) {
    return std::any_of(candidates.begin(), candidates.end(),
                       [id](const auto& candidate) { return candidate.id == id; });
}

const ContextRetentionPolicy::Candidate* find(
    const std::vector<ContextRetentionPolicy::Candidate>& candidates, std::uint64_t id) {
    for (const auto& candidate : candidates) {
        if (candidate.id == id) { return &candidate; }
    }
    return nullptr;
}

int test_session_lineage_and_candidates() {
    int failures = 0;
    ContextRetentionPolicy policy;
    const ContinuationOwnerToken a = policy.open_owner(std::string("session-1"));
    const ContinuationOwnerToken c = policy.open_owner(std::string("session-2"));
    const ContinuationOwnerToken d = policy.open_owner(std::string("session-1"));
    policy.publish(a, 1, CheckpointRole::InputReplay, /*shared=*/false);
    policy.publish(a, 2, CheckpointRole::SharedPrefix, /*shared=*/true);
    policy.publish(c, 3, CheckpointRole::InputReplay, /*shared=*/false);

    const auto own = policy.candidates(a, std::string_view("session-1"));
    failures += check(has(own, 1) && has(own, 2) && !has(own, 3),
                      "an owner sees its private and every shared checkpoint");

    const auto other_active = policy.candidates(d, std::string_view("session-1"));
    failures += check(!has(other_active, 1) && has(other_active, 2),
                      "an active owner's private checkpoint stays private");

    policy.finish_owner(a);
    failures += check(policy.session_owner("session-1") == std::optional<ContinuationOwnerToken>(a),
                      "finishing an owner advances the session lineage");
    const auto after_finish = policy.candidates(d, std::string_view("session-1"));
    failures += check(has(after_finish, 1), "a finished session owner's checkpoint is reusable");
    const auto* reused = find(after_finish, 1);
    failures += check(reused != nullptr && reused->session_hint,
                      "a same-session checkpoint carries the session hint");
    const auto other_session = policy.candidates(d, std::string_view("session-2"));
    failures += check(!has(other_session, 1) && has(other_session, 2),
                      "a different session cannot restore the checkpoint");

    // Abandoning an owner drops its private checkpoints but not the shared one.
    policy.abandon_owner(a);
    failures += check(!policy.contains(1) && policy.contains(2),
                      "abandon removes private checkpoints and keeps shared ones");
    return failures;
}

int test_victims_and_admits() {
    int failures = 0;
    ContextRetentionPolicy policy;
    const ContinuationOwnerToken owner = policy.open_owner();
    policy.publish(owner, 10, CheckpointRole::SharedPrefix, true);
    policy.publish(owner, 11, CheckpointRole::SharedPrefix, true);

    // Never-reused victims come first, oldest publication first.
    std::vector<std::uint64_t> victims = policy.victims();
    failures += check(victims.size() == 2 && victims[0] == 10 && victims[1] == 11,
                      "fresh checkpoints are evicted oldest first");

    policy.remember_demand(10, 100);
    victims = policy.victims();
    failures += check(victims[0] == 11, "a never-reused checkpoint is evicted before a reused one");

    policy.remember_demand(11, 50);
    victims = policy.victims();
    failures += check(victims[0] == 11 && victims[1] == 10,
                      "reused checkpoints are evicted least-recently-demanded first");

    failures += check(policy.priority_of(10).reused && policy.priority_of(10).last_demand == 100,
                      "demand priority is recorded");
    failures += check(!policy.priority_of(999).reused, "unknown checkpoints have no demand");

    policy.publish(owner, 12, CheckpointRole::SharedPrefix, true);
    const std::vector<std::uint64_t> fresh{12};
    failures += check(!policy.admits(1, fresh) && policy.admits(2, fresh),
                      "a fresh checkpoint admits only above its loss");
    failures += check(!policy.admits(4, std::vector<std::uint64_t>{10}) &&
                          policy.admits(5, std::vector<std::uint64_t>{10}),
                      "a reused checkpoint admits only above the reused loss");
    failures += check(!policy.admits(8, std::vector<std::uint64_t>{10, 11}) &&
                          policy.admits(9, std::vector<std::uint64_t>{10, 11}),
                      "replacing two checkpoints needs their combined loss");
    return failures;
}

// A newer checkpoint of the same session supersedes the older one, which makes it the first
// victim and makes reclaiming it free. Both residencies of one request stay, because they share
// an owner, and an active or different-session lineage is left alone.
int test_superseded_within_session() {
    int failures = 0;
    ContextRetentionPolicy policy;
    const ContinuationOwnerToken first = policy.open_owner(std::string("session-1"));
    policy.publish(first, 20, CheckpointRole::SharedPrefix, true);
    policy.publish(first, 21, CheckpointRole::SharedPrefix, true);
    failures += check(!policy.superseded(20) && !policy.superseded(21),
                      "two residencies of one request do not supersede each other");

    const ContinuationOwnerToken other = policy.open_owner(std::string("session-2"));
    policy.publish(other, 23, CheckpointRole::SharedPrefix, true);
    failures += check(!policy.superseded(20),
                      "a checkpoint of another session does not supersede");

    const ContinuationOwnerToken second = policy.open_owner(std::string("session-1"));
    policy.publish(second, 22, CheckpointRole::SharedPrefix, true);
    failures += check(!policy.superseded(20),
                      "an active same-session lineage does not supersede while it may continue");

    policy.finish_owner(first);
    failures += check(policy.superseded(20) && policy.superseded(21),
                      "a finished session's checkpoints are superseded by the newer owner");
    failures += check(!policy.superseded(22) && !policy.superseded(23),
                      "the superseding checkpoint and other sessions stay intact");

    // A superseded checkpoint is reclaimed before fresh ones even when it was reused.
    policy.remember_demand(20, 100);
    const std::vector<std::uint64_t> victims = policy.victims();
    failures += check(victims.size() >= 2 && victims[0] == 21 && victims[1] == 20,
                      "superseded checkpoints are evicted first, freshest of them first");

    // Reclaiming a superseded checkpoint costs nothing: the newer prefix still offers it, so an
    // admission that would lose against a reused checkpoint succeeds against a superseded one.
    const std::vector<std::uint64_t> superseded_ten{20};
    failures += check(policy.admits(1, superseded_ten),
                      "reclaiming a superseded checkpoint costs no reuse");
    const std::vector<std::uint64_t> reused_not_superseded{23};
    failures += check(!policy.admits(1, reused_not_superseded),
                      "a non-superseded checkpoint keeps its loss");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_session_lineage_and_candidates();
    failures += test_victims_and_admits();
    failures += test_superseded_within_session();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
