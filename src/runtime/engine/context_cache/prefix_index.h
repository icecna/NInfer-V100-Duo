#pragma once

// Engine-level shortlist index over checkpoint keys. A checkpoint is identified by
// (identity_tag, frontier, digest); the index answers "which stored checkpoints could serve this
// prompt" by comparing the incoming key at each stored frontier. The authoritative token/media
// comparison runs after the index returns its candidates, so a candidate is only a lead.
//
// The index owns handles, not checkpoint state; the owning program keeps the state image and KV.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::runtime {

template <class Handle, class Key>
class PrefixIndex {
public:
    void insert(Handle handle, Key key) {
        for (Entry& entry : entries_) {
            if (entry.handle == handle) {
                entry.key = key;
                return;
            }
        }
        entries_.push_back(Entry{handle, key});
    }

    void erase(Handle handle) {
        entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
                                      [&](const Entry& entry) { return entry.handle == handle; }),
                       entries_.end());
    }

    void clear() noexcept { entries_.clear(); }

    [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }
    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }

    [[nodiscard]] bool contains(Handle handle) const {
        for (const Entry& entry : entries_) {
            if (entry.handle == handle) { return true; }
        }
        return false;
    }

    // Candidates whose frontier is at or below `limit` and whose stored key equals the incoming key
    // for that frontier, deepest first. Equal frontiers keep insertion order.
    template <class KeyAt>
    [[nodiscard]] std::vector<Handle> matches(std::uint32_t limit, KeyAt&& key_at) const {
        std::vector<Candidate> candidates;
        for (const Entry& entry : entries_) {
            if (entry.key.frontier == 0 || entry.key.frontier > limit) { continue; }
            if (!(key_at(entry.key.frontier) == entry.key)) { continue; }
            candidates.push_back(Candidate{entry.key.frontier, &entry});
        }
        std::stable_sort(candidates.begin(), candidates.end(),
                         [](const Candidate& left, const Candidate& right) {
                             return left.frontier > right.frontier;
                         });
        std::vector<Handle> handles;
        handles.reserve(candidates.size());
        for (const Candidate& candidate : candidates) { handles.push_back(candidate.entry->handle); }
        return handles;
    }

private:
    struct Entry {
        Handle handle;
        Key key;
    };
    struct Candidate {
        std::uint32_t frontier = 0;
        const Entry* entry     = nullptr;
    };

    std::vector<Entry> entries_;
};

} // namespace ninfer::runtime
