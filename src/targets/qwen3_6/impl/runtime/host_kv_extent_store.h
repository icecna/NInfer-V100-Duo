#pragma once

// Host-resident replicas of KV pages, one extent per checkpoint prefix. The store owns the arena
// allocation and the ordered logical-page membership; the page registry owns the per-page facts.
// Reserving pins the sources so the device pages cannot change under the copy, publishing attaches
// the host replica to each page and releases the pins, and an extent whose pages are all
// unreferenced is released back to the arena.
//
// All operations are caller-serialized (the executor thread); there is no locking.

#include "core/host_kv_arena.h"
#include "targets/qwen3_6/impl/runtime/kv_page_registry.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

class HostKvExtentStore {
public:
    // RAII handle for a reserved extent whose sources are pinned. Destroying it without publish()
    // aborts the reservation.
    class Reservation {
    public:
        Reservation() noexcept = default;
        ~Reservation();
        Reservation(Reservation&& other) noexcept;
        Reservation& operator=(Reservation&& other) noexcept;
        Reservation(const Reservation&)            = delete;
        Reservation& operator=(const Reservation&) = delete;

        [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    private:
        friend class HostKvExtentStore;
        Reservation(HostKvExtentStore& owner, std::uint32_t index, std::uint32_t generation) noexcept
            : owner_(&owner), index_(index), generation_(generation) {}

        HostKvExtentStore* owner_  = nullptr;
        std::uint32_t index_       = 0;
        std::uint32_t generation_  = 0;
    };

    // RAII handle for a published extent whose pages carry a host replica.
    class Capability {
    public:
        Capability() noexcept = default;
        ~Capability();
        Capability(Capability&& other) noexcept;
        Capability& operator=(Capability&& other) noexcept;
        Capability(const Capability&)            = delete;
        Capability& operator=(const Capability&) = delete;

        [[nodiscard]] bool valid() const noexcept { return owner_ != nullptr; }

    private:
        friend class HostKvExtentStore;
        Capability(HostKvExtentStore& owner, std::uint32_t index, std::uint32_t generation) noexcept
            : owner_(&owner), index_(index), generation_(generation) {}

        HostKvExtentStore* owner_  = nullptr;
        std::uint32_t index_       = 0;
        std::uint32_t generation_  = 0;
    };

    HostKvExtentStore(HostKVArena& arena, KvPageRegistry& registry,
                      std::uint32_t descriptor_capacity = 16);

    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(extents_.size());
    }
    [[nodiscard]] std::uint32_t occupied() const noexcept { return occupied_; }
    [[nodiscard]] std::size_t occupied_bytes() const noexcept { return occupied_bytes_; }

    // Reserve host storage for `pages` in order and pin their sources. Fails when a page cannot be
    // pinned or the arena cannot satisfy the request.
    [[nodiscard]] std::optional<Reservation> reserve(const HostKVPageLayout& layout,
                                                     std::span<const std::uint32_t> pages);
    // Attach the host replica to every page and release the source pins.
    [[nodiscard]] std::optional<Capability> publish(Reservation&& reservation);
    void abort(Reservation&& reservation) noexcept;
    void release(Capability&& capability) noexcept;

    // Release every published extent whose pages are all unreferenced and unpinned.
    void release_unreferenced();

    [[nodiscard]] HostKVAllocationView view(const Capability& capability) const;

private:
    enum class State : std::uint8_t { Free, Reserved, Published };

    struct Extent {
        HostKVAllocation allocation;
        std::vector<std::uint32_t> pages;
        std::uint32_t generation = 1;
        std::size_t bytes        = 0;
        State state              = State::Free;
    };

    [[nodiscard]] Extent& require(std::uint32_t index, std::uint32_t generation);
    void abort(std::uint32_t index, std::uint32_t generation) noexcept;
    void release(std::uint32_t index, std::uint32_t generation) noexcept;
    [[nodiscard]] bool page_releasable(std::uint32_t page) const;

    HostKVArena* arena_      = nullptr;
    KvPageRegistry* registry_ = nullptr;
    std::vector<Extent> extents_;
    std::uint32_t occupied_       = 0;
    std::size_t occupied_bytes_   = 0;
};

} // namespace ninfer::targets::qwen3_6::detail
