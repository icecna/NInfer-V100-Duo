#pragma once

// Retention policy for published context-cache checkpoints. It owns the semantic bookkeeping the
// executor needs to decide what a request may reuse and what may be reclaimed:
//
//   owners       a sequence that publishes checkpoints; carries an optional session key
//   checkpoints  a shared or owner-private checkpoint with its role and publication order
//   demand       the last request that reused a checkpoint, with a bounded LRU
//
// The policy owns no checkpoint state. It is an engine-level, host-only primitive; the executor
// (L5d) drives it and the owning program keeps the KV/state images.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace ninfer::runtime {

using ContinuationOwnerToken = std::uint64_t;
using CheckpointId           = std::uint64_t;

enum class CheckpointRole : std::uint8_t {
    InputReplay,
    LongAnchor,
    SharedPrefix,
    Continuation,
};

struct RetentionPriority {
    bool reused               = false;
    std::uint64_t last_demand = 0;
};

class ContextRetentionPolicy {
public:
    static constexpr std::size_t kDemandCapacity = 256;
    // Reclaiming a checkpoint some request recently reused is treated as this much more costly
    // than reclaiming one that was never reused.
    static constexpr std::uint64_t kReusedLoss = 4;
    static constexpr std::uint64_t kFreshLoss  = 1;

    struct Candidate {
        CheckpointId id             = 0;
        CheckpointRole role         = CheckpointRole::SharedPrefix;
        bool shared                 = false;
        bool session_hint           = false;
    };

    // --- owner lifecycle ---

    ContinuationOwnerToken open_owner(std::optional<std::string> session_key = std::nullopt) {
        const ContinuationOwnerToken token = next_owner_++;
        owners_[token] = Owner{.session = std::move(session_key), .active = true};
        return token;
    }

    // Mark the owner finished and make it the newest owner of its session lineage.
    void finish_owner(ContinuationOwnerToken owner) {
        auto it = owners_.find(owner);
        if (it == owners_.end()) { return; }
        it->second.active = false;
        if (it->second.session) { sessions_[*it->second.session] = owner; }
    }

    void abandon_owner(ContinuationOwnerToken owner) {
        auto it = owners_.find(owner);
        if (it == owners_.end()) { return; }
        for (auto checkpoint = checkpoints_.begin(); checkpoint != checkpoints_.end();) {
            if (checkpoint->second.owner == owner && !checkpoint->second.shared) {
                forget(checkpoint->first);
                checkpoint = checkpoints_.erase(checkpoint);
            } else {
                ++checkpoint;
            }
        }
        owners_.erase(it);
    }

    [[nodiscard]] bool owner_active(ContinuationOwnerToken owner) const {
        const auto it = owners_.find(owner);
        return it != owners_.end() && it->second.active;
    }

    [[nodiscard]] std::optional<std::string> session_of(ContinuationOwnerToken owner) const {
        const auto it = owners_.find(owner);
        return it != owners_.end() ? it->second.session : std::nullopt;
    }

    [[nodiscard]] std::optional<ContinuationOwnerToken> session_owner(std::string_view session) const {
        const auto it = sessions_.find(std::string(session));
        return it != sessions_.end() ? std::optional<ContinuationOwnerToken>(it->second)
                                     : std::nullopt;
    }

    [[nodiscard]] std::optional<std::string> checkpoint_session(CheckpointId id) const {
        const auto checkpoint = checkpoints_.find(id);
        if (checkpoint == checkpoints_.end()) { return std::nullopt; }
        const auto owner = owners_.find(checkpoint->second.owner);
        return owner != owners_.end() ? owner->second.session : std::nullopt;
    }

    // Whether `id` may be offered to a request: shared prefixes are always visible, anonymous
    // prefixes are visible to anyone, and a named prefix is visible only to the requesting owner
    // itself or to a finished owner of the same session.
    [[nodiscard]] bool visible(CheckpointId id, ContinuationOwnerToken own_owner,
                               std::optional<std::string_view> session) const {
        const auto checkpoint = checkpoints_.find(id);
        if (checkpoint == checkpoints_.end()) { return false; }
        if (checkpoint->second.shared || checkpoint->second.owner == own_owner) { return true; }
        const auto owner = owners_.find(checkpoint->second.owner);
        if (owner == owners_.end()) { return false; }
        if (!owner->second.session) { return true; }
        if (owner->second.active || !session) { return false; }
        return *owner->second.session == *session;
    }

    // --- checkpoints ---

    void publish(ContinuationOwnerToken owner, CheckpointId id, CheckpointRole role, bool shared) {
        checkpoints_[id] = Checkpoint{.owner       = owner,
                                      .role        = role,
                                      .shared      = shared,
                                      .publication = ++clock_,
                                      .retained_at = ++clock_};
    }

    void erase(CheckpointId id) {
        forget(id);
        checkpoints_.erase(id);
    }

    [[nodiscard]] bool contains(CheckpointId id) const {
        return checkpoints_.find(id) != checkpoints_.end();
    }

    [[nodiscard]] std::size_t size() const noexcept { return checkpoints_.size(); }

    // --- demand ---

    void remember_demand(CheckpointId id, std::uint64_t request_id) {
        if (!contains(id)) { return; }
        demand_[id] = request_id;
        demand_lru_.erase(std::remove(demand_lru_.begin(), demand_lru_.end(), id),
                          demand_lru_.end());
        demand_lru_.push_back(id);
        while (demand_lru_.size() > kDemandCapacity) {
            demand_.erase(demand_lru_.front());
            demand_lru_.erase(demand_lru_.begin());
        }
    }

    [[nodiscard]] RetentionPriority priority_of(CheckpointId id) const {
        const auto it = demand_.find(id);
        return RetentionPriority{.reused = it != demand_.end(),
                                 .last_demand = it != demand_.end() ? it->second : 0};
    }

    // --- candidates ---

    // Shared checkpoints plus private checkpoints the requesting lineage may restore: its own, or
    // one from a finished owner of the same session when the request names that session.
    [[nodiscard]] std::vector<Candidate> candidates(
        ContinuationOwnerToken own_owner,
        std::optional<std::string_view> session = std::nullopt) const {
        std::vector<Candidate> result;
        for (const auto& [id, checkpoint] : checkpoints_) {
            const auto owner = owners_.find(checkpoint.owner);
            const bool owner_session_matches =
                owner != owners_.end() && owner->second.session && session &&
                *owner->second.session == *session;
            const bool usable =
                checkpoint.shared || checkpoint.owner == own_owner ||
                (owner != owners_.end() && !owner->second.active && owner_session_matches);
            if (!usable) { continue; }
            result.push_back(Candidate{.id       = id,
                                       .role     = checkpoint.role,
                                       .shared   = checkpoint.shared,
                                       .session_hint = owner_session_matches});
        }
        return result;
    }

    // --- reclaim ---

    // Least valuable first: never-reused checkpoints by age, then reused ones by last demand.
    [[nodiscard]] std::vector<CheckpointId> victims() const {
        std::vector<CheckpointId> ids;
        ids.reserve(checkpoints_.size());
        for (const auto& [id, checkpoint] : checkpoints_) { ids.push_back(id); }
        const auto rank = [this](CheckpointId id) {
            const RetentionPriority priority = priority_of(id);
            return std::pair<bool, std::uint64_t>{
                priority.reused,
                priority.reused ? priority.last_demand : checkpoints_.at(id).retained_at};
        };
        std::stable_sort(ids.begin(), ids.end(), [&](CheckpointId left, CheckpointId right) {
            return rank(left) < rank(right);
        });
        return ids;
    }

    // A new checkpoint with `gain` may replace the sacrificed ones only when its gain exceeds the
    // combined loss of reclaiming them.
    [[nodiscard]] bool admits(std::uint64_t gain,
                              std::span<const CheckpointId> sacrificed) const {
        std::uint64_t loss = 0;
        for (const CheckpointId id : sacrificed) {
            loss += priority_of(id).reused ? kReusedLoss : kFreshLoss;
        }
        return gain > loss;
    }

private:
    struct Owner {
        std::optional<std::string> session;
        bool active = true;
    };
    struct Checkpoint {
        ContinuationOwnerToken owner = 0;
        CheckpointRole role          = CheckpointRole::SharedPrefix;
        bool shared                  = false;
        std::uint64_t publication    = 0;
        std::uint64_t retained_at    = 0;
    };

    void forget(CheckpointId id) {
        demand_.erase(id);
        demand_lru_.erase(std::remove(demand_lru_.begin(), demand_lru_.end(), id),
                          demand_lru_.end());
    }

    std::unordered_map<ContinuationOwnerToken, Owner> owners_;
    std::unordered_map<CheckpointId, Checkpoint> checkpoints_;
    std::unordered_map<std::string, ContinuationOwnerToken> sessions_;
    std::unordered_map<CheckpointId, std::uint64_t> demand_;
    std::vector<CheckpointId> demand_lru_;
    std::uint64_t clock_     = 0;
    ContinuationOwnerToken next_owner_ = 1;
};

} // namespace ninfer::runtime
