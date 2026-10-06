#pragma once

// HostPrefixTable: storage container for host-resident prefix snapshots.
//
// The table stores completed D2H prefix snapshots keyed by a caller-supplied id, not by content.
// Matching, session visibility, and retention policy live elsewhere: the Program keeps a
// PrefixIndex over the snapshots' shortlist keys and a ContextRetentionPolicy over their
// owners/lineages. The table owns only membership, the LRU stamp bookkeeping, and the in-flight
// claim guard that keeps a snapshot's extents alive while a reader loads them.

#include <cstddef>
#include <cstdint>
#include <limits>
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
    // Monotonic LRU order used by the retention policy's age signal. Zero until first use.
    std::uint64_t lru_stamp = 0;
    // Number of in-flight H2D claims reading this entry's extents. Nonzero entries are
    // unevictable until the claims complete (the reader holds the allocations).
    std::uint32_t in_flight_claims = 0;
};

// Marker callback type meaning "no evict notification"; the default for HostPrefixTable::evict.
struct NoHostPrefixEvictionCallback {};

template <class Payload>
class HostPrefixTable {
public:
    // Insert a fresh entry and return its id. `id_provider` is any zero-argument callable that
    // yields an id the table has never handed out (production uses a process-wide counter).
    template <class IdProvider>
    HostPrefixId insert(Payload payload, IdProvider&& id_provider)
    {
        HostPrefixId id = id_provider();
        if (id == kInvalidHostPrefixId) { id = next_synthetic_id_; }
        entries_.push_back(HostPrefixEntry{
            .id       = id,
            .payload  = std::move(payload),
            .lru_stamp = 0,
            .in_flight_claims = 0,
        });
        if (id < next_synthetic_id_) { next_synthetic_id_ = id + 1; }
        return id;
    }

    // Assign an entry's LRU stamp (the production route stamps when its D2H completes or a
    // host entry is re-used). Unknown ids are a no-op.
    void stamp(HostPrefixId id, std::uint64_t lru_stamp) {
        if (auto* entry = find(id)) { entry->lru_stamp = lru_stamp; }
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
