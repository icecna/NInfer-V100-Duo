#pragma once

// HostPrefixTable: the content-addressed, lane-free resident prefix table (P1 core).
//
// The table stores completed D2H prefix snapshots keyed by their content, not by the lane that
// produced them. Entries carry everything the reuse planner and the claim path need; the
// payload type is chosen by the caller so this table stays a pure CPU data structure with no
// CUDA, pool, or model dependencies. The caller owns entry lifecycle policy (capacity checks
// against the host arena, device work); the table owns membership, LRU order, and the
// in-flight claim guard.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

// Identity of one table entry. Stable across the entry's whole lifetime; used by callers to
// invalidate plans that reference the entry after it is evicted.
using HostPrefixId = std::uint64_t;

inline constexpr HostPrefixId kInvalidHostPrefixId =
    std::numeric_limits<HostPrefixId>::max();

template <class Payload>
struct HostPrefixEntry {
    HostPrefixId id = kInvalidHostPrefixId;
    Payload payload;
    // Monotonic LRU order; smallest is the eviction victim. Zero until the first use; entries
    // with no stamp yet are never victims.
    std::uint64_t lru_stamp = 0;
    // Session lineage that produced the entry; empty means anonymous (reusable by anyone).
    std::string session_key;
    // Last request that reused the entry; zero until the first reuse. A demanded entry is a less
    // attractive eviction victim than a never-demanded one.
    std::uint64_t demand_stamp = 0;
    // Number of in-flight H2D claims reading this entry's extents. Nonzero entries are
    // unevictable until the claims complete (the reader holds the allocations).
    std::uint32_t in_flight_claims = 0;
};

// Prompt-shaped input the identity is matched against; the concrete type is caller-defined
// (the production route will pass the prepared prompt).
template <class Payload>
using HostPrefixPrompt = typename Payload::Prompt;

// Marker callback type meaning "no evict notification"; the default for HostPrefixTable::evict.
struct NoHostPrefixEvictionCallback {};

template <class Payload>
class HostPrefixTable {
public:
    // Insert a fresh entry and return its id. `id_provider` is any zero-argument callable that
    // yields an id the table has never handed out (production uses a process-wide counter).
    // `session_key` names the lineage that produced the entry; empty means anonymous.
    template <class IdProvider>
    HostPrefixId insert(Payload payload, IdProvider&& id_provider, std::string session_key = {})
    {
        HostPrefixId id = id_provider();
        if (id == kInvalidHostPrefixId) { id = next_synthetic_id_; }
        entries_.push_back(HostPrefixEntry{
            .id       = id,
            .payload  = std::move(payload),
            .lru_stamp = 0,
            .session_key = std::move(session_key),
            .demand_stamp = 0,
            .in_flight_claims = 0,
        });
        if (id < next_synthetic_id_) { next_synthetic_id_ = id + 1; }
        return id;
    }

    // Assign an entry's LRU stamp (the production route stamps when its D2H completes or a
    // host entry is re-used). Unknown ids are a no-op.
    void stamp(HostPrefixId id, std::uint64_t lru_stamp) { if (auto* e = find(id)) { e->lru_stamp = lru_stamp; } }

    // Record that `id` was reused by `demand_stamp`; a demanded entry resists eviction.
    void mark_demand(HostPrefixId id, std::uint64_t demand_stamp) {
        if (auto* e = find(id)) { e->demand_stamp = demand_stamp; }
    }

    const HostPrefixEntry<Payload>* find(HostPrefixId id) const noexcept
    {
        for (const auto& entry : entries_) {
            if (entry.id == id) { return &entry; }
        }
        return nullptr;
    }

    [[nodiscard]] HostPrefixEntry<Payload>* find(HostPrefixId id) noexcept
    {
        for (auto& entry : entries_) {
            if (entry.id == id) { return &entry; }
        }
        return nullptr;
    }

    // Content lookup: `depth_of(prompt, payload)` returns the entry's usable match depth (0 =
    // no match). The DEEPEST match wins -- not the largest stored boundary: a prompt that
    // fails the deep identity check may still share a short token prefix with several entries,
    // and picking by stored boundary would select an unrelated long entry and claim it for a
    // trivial prefix. Ties keep the first inserted. A use refreshes the LRU stamp (claiming
    // counts as use); `on_use` supplies the caller's monotonic counter.
    //
    // When `session_key` is non-empty, anonymous entries and entries from the same lineage stay
    // eligible; an entry published by a different lineage is skipped entirely.
    template <class DepthOf, class OnUse>
    HostPrefixEntry<Payload>* find_best(const HostPrefixPrompt<Payload>& prompt,
                                        DepthOf depth_of, OnUse on_use,
                                        std::string_view session_key = {})
    {
        HostPrefixEntry<Payload>* best = nullptr;
        std::size_t best_depth = 0;
        bool best_session = false;
        for (auto& entry : entries_) {
            const bool session_match = !session_key.empty() && entry.session_key == session_key;
            if (!session_key.empty() && !entry.session_key.empty() && !session_match) { continue; }
            const std::size_t depth = depth_of(prompt, entry.payload);
            if (depth == 0) { continue; }
            if (best == nullptr || depth > best_depth ||
                (depth == best_depth && session_match && !best_session)) {
                best = &entry;
                best_depth = depth;
                best_session = session_match;
            }
        }
        if (best != nullptr) { best->lru_stamp = on_use(); }
        return best;
    }

    // Oldest evictable entry (in_flight_claims == 0). A never-demanded entry is a more
    // attractive victim than a demanded one; within a class the smallest demand (or LRU) stamp
    // wins. Never-stamped entries are unevictable except when no stamped evictable entry exists.
    const HostPrefixEntry<Payload>* oldest_evictable() const noexcept
    {
        const HostPrefixEntry<Payload>* victim = nullptr;
        bool victim_unstamped = false;
        for (const auto& entry : entries_) {
            if (entry.in_flight_claims != 0) { continue; }
            if (victim == nullptr) {
                victim = &entry;
                victim_unstamped = entry.lru_stamp == 0 && entry.demand_stamp == 0;
                continue;
            }
            const bool entry_unstamped = entry.lru_stamp == 0 && entry.demand_stamp == 0;
            if (victim_unstamped != entry_unstamped) {
                if (victim_unstamped) {
                    victim = &entry;
                    victim_unstamped = false;
                }
                continue;
            }
            if (victim_unstamped) { continue; }
            const auto rank = [](const HostPrefixEntry<Payload>& e) {
                return e.demand_stamp != 0 ? e.demand_stamp : e.lru_stamp;
            };
            if (rank(entry) < rank(*victim)) { victim = &entry; }
        }
        return victim;
    }

    // Remove an entry. `on_evict` (when provided) is invoked before removal so the caller can
    // invalidate dependent plans. Returns false for unknown ids.
    template <class OnEvict = NoHostPrefixEvictionCallback>
    bool evict(HostPrefixId id, OnEvict on_evict = {})
    {
        if constexpr (!std::is_same_v<OnEvict, NoHostPrefixEvictionCallback>) { on_evict(id); }
        auto it = entries_.begin();
        while (it != entries_.end()) {
            if (it->id == id) { entries_.erase(it); return true; }
            ++it;
        }
        return false;
    }

    void begin_claim(HostPrefixId id) noexcept
    {
        if (auto* entry = find(id)) { entry->in_flight_claims++; }
    }

    void end_claim(HostPrefixId id) noexcept
    {
        if (auto* entry = find(id)) {
            if (entry->in_flight_claims == 0) { return; }
            entry->in_flight_claims--;
        }
    }

    std::size_t size() const noexcept { return entries_.size(); }
    bool empty() const noexcept { return entries_.empty(); }
    std::size_t in_flight_claims() const noexcept
    {
        std::size_t total = 0;
        for (const auto& entry : entries_) { total += entry.in_flight_claims; }
        return total;
    }
    void clear() noexcept { entries_.clear(); }

private:
    std::vector<HostPrefixEntry<Payload>> entries_;
    HostPrefixId next_synthetic_id_ = 0;
};

} // namespace ninfer::targets::qwen3_6::detail