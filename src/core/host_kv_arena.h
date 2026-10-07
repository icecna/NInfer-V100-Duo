#pragma once

#include "core/arena.h"
#include "core/dtype.h"
#include "core/paged_kv_cache.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer {

// One plane of the canonical HOST representation of a Paged KV pool page. The host copy is the
// device plane's payload packed at 256-byte boundaries: for a PageMajor device plane the whole
// page is one contiguous run (page_payload_bytes). Only PageMajor device pools are supported
// end to end: the Program's D2H/H2D copy paths reject any other plane order, and the
// head_extent/head_payload_bytes fields exist for the layout's descriptive completeness.
struct HostKVPlaneLayout {
    DType dtype                = DType::BF16;
    std::int32_t leading_extent = 0;
    std::int32_t head_extent    = 0;
    std::size_t offset             = 0;
    std::size_t page_payload_bytes = 0;
    std::size_t head_payload_bytes = 0;

    [[nodiscard]] friend bool operator==(const HostKVPlaneLayout&,
                                         const HostKVPlaneLayout&) = default;
};

// The host geometry of one pool: which planes, in what order, and the stride between logical
// page records. Identity is (device_plane_order, planes); two pools with identical plane
// geometry share one layout entry in the owning arena.
struct HostKVPageLayout {
    PagedKVPlaneOrder device_plane_order = PagedKVPlaneOrder::PageMajor;
    std::vector<HostKVPlaneLayout> planes;
    std::size_t page_stride = 0;

    [[nodiscard]] friend bool operator==(const HostKVPageLayout&, const HostKVPageLayout&) =
        default;
};

// Derives the canonical host page layout for a pool. The pool's plane specs fully determine the
// result; page capacity and block-table rows are irrelevant to the host record.
[[nodiscard]] HostKVPageLayout plan_host_kv_page_layout(const PagedKVPoolSpec& spec);

class HostKVArena;

// Non-owning access to one arena allocation's host pages: a base pointer into the arena's pinned
// backing plus the page layout. Writes through data() are the D2H destination; reads are the H2D
// source. The view is invalidated by any arena mutation (bump of the arena revision).
class HostKVAllocationView {
public:
    HostKVAllocationView() noexcept = default;

    [[nodiscard]] bool valid() const noexcept;

    [[nodiscard]] std::byte* data() const noexcept { return data_; }

    [[nodiscard]] std::uint32_t page_count() const noexcept { return page_count_; }

    // Byte count of a raw (non-page) extent; zero for page-run views.
    [[nodiscard]] std::size_t raw_bytes() const noexcept { return raw_bytes_; }

    [[nodiscard]] const HostKVPageLayout& layout() const;

    // View over a sub-run of logical pages; the base pointer lands on the page record.
    [[nodiscard]] HostKVAllocationView subview(std::uint32_t begin, std::uint32_t count) const;

private:
    friend class HostKVArena;

    HostKVAllocationView(const HostKVArena* arena, std::uint64_t revision, std::byte* data,
                         const HostKVPageLayout* layout, std::uint32_t page_count) noexcept
        : arena_(arena), revision_(revision), data_(data), layout_(layout),
          page_count_(page_count) {}

    HostKVAllocationView(const HostKVArena* arena, std::uint64_t revision, std::byte* data,
                         std::size_t raw_bytes) noexcept
        : arena_(arena), revision_(revision), data_(data), raw_bytes_(raw_bytes) {}

    const HostKVArena* arena_ = nullptr;
    std::uint64_t revision_   = 0;
    std::byte* data_        = nullptr;
    const HostKVPageLayout* layout_ = nullptr;
    std::uint32_t page_count_       = 0;
    std::size_t raw_bytes_          = 0;
};

// Owning handle to a host allocation inside a HostKVArena. Destructor (or release) returns the
// extent; move-only, non-copyable.
class HostKVAllocation {
public:
    HostKVAllocation() noexcept = default;
    ~HostKVAllocation();

    HostKVAllocation(const HostKVAllocation&)            = delete;
    HostKVAllocation& operator=(const HostKVAllocation&) = delete;
    HostKVAllocation(HostKVAllocation&& other) noexcept;
    HostKVAllocation& operator=(HostKVAllocation&& other) noexcept;

    [[nodiscard]] bool valid() const noexcept { return arena_ != nullptr; }

    [[nodiscard]] std::uint32_t page_count() const noexcept;

    // Byte count of a raw (non-page) extent; zero for page runs.
    [[nodiscard]] std::size_t raw_bytes() const noexcept;

    bool release() noexcept;

private:
    friend class HostKVArena;

    HostKVAllocation(HostKVArena& owner, std::uint32_t descriptor, std::uint32_t generation)
        noexcept
        : arena_(&owner), descriptor_(descriptor), generation_(generation) {}

    void disarm() noexcept;

    HostKVArena* arena_      = nullptr;
    std::uint32_t descriptor_ = 0;
    std::uint32_t generation_ = 0;
};

// One region of an ordered host allocation: either a closed-set page layout plus a logical page
// count, or (raw_bytes > 0) a plain byte extent with no page structure. The two forms are
// mutually exclusive.
struct HostKVAllocationPlan {
    const HostKVPageLayout* layout = nullptr;
    std::uint32_t pages            = 0;
    std::size_t raw_bytes           = 0;
};

// Fixed-capacity arena of pinned host memory handing out logical-page runs for a closed set of
// pool page layouts. Allocation is first-fit over coalesced free extents; release merges. This is
// the per-device home of retained-prefix KV that has been evicted from the device pool.
//
// All operations are caller-serialized (the executor thread); there is no locking.
class HostKVArena {
public:
    HostKVArena(std::size_t capacity_bytes, std::span<const HostKVPageLayout> supported_layouts);

    HostKVArena(const HostKVArena&)            = delete;
    HostKVArena& operator=(const HostKVArena&) = delete;
    HostKVArena(HostKVArena&&)                 = delete;
    HostKVArena& operator=(HostKVArena&&)      = delete;

    [[nodiscard]] std::size_t capacity_bytes() const noexcept { return capacity_bytes_; }

    [[nodiscard]] std::size_t occupied_bytes() const noexcept { return occupied_bytes_; }

    // The largest single free run. A new allocation has to fit in one run, so this is the number
    // that separates "full" from "fragmented"; upstream never compacts, so it is the quantity to
    // watch before deciding to quantize allocations or to page-list the extents.
    [[nodiscard]] std::size_t largest_free_extent_bytes() const noexcept {
        std::size_t largest = 0;
        for (const FreeExtent& extent : free_extents_) {
            if (extent.bytes > largest) { largest = extent.bytes; }
        }
        return largest;
    }

    [[nodiscard]] std::size_t free_bytes() const noexcept {
        return capacity_bytes_ - occupied_bytes_;
    }

    [[nodiscard]] const HostKVPageLayout* layout_for(const PagedKVPoolSpec& spec) const noexcept;

    [[nodiscard]] bool can_allocate(const HostKVPageLayout& layout, std::uint32_t pages)
        const noexcept;

    // Sequential feasibility: true only if the arena would place every plan entry in order,
    // consuming one contiguous free extent per entry exactly as a same-order run of allocate()
    // calls on the unchanged arena would. Per-entry can_allocate() over-approximates when the
    // entries' combined size exceeds the largest free extent while each entry alone fits.
    [[nodiscard]] bool can_allocate_all(std::span<const HostKVAllocationPlan> plan) const
        noexcept;

    // Fails (nullopt) when capacity or the closed layout set does not admit the request; the
    // arena is left unchanged.
    [[nodiscard]] std::optional<HostKVAllocation> allocate(const HostKVPageLayout& layout,
                                                           std::uint32_t pages) noexcept;

    // Raw (non-page) extent: a contiguous run of pinned bytes (256-byte aligned) sharing the
    // same budget and free-extent pool as the page runs. Fails (nullopt) when no free extent
    // admits the request; the arena is left unchanged.
    [[nodiscard]] std::optional<HostKVAllocation> allocate_raw(std::size_t bytes) noexcept;

    [[nodiscard]] HostKVAllocationView view(const HostKVAllocation& allocation) const;

    // View over a raw extent's bytes; throws for a page-run allocation.
    [[nodiscard]] HostKVAllocationView raw_view(const HostKVAllocation& allocation) const;

private:
    friend class HostKVAllocation;
    friend class HostKVAllocationView;

    struct Descriptor {
        std::size_t offset       = 0;
        std::size_t bytes        = 0;
        std::uint32_t layout     = 0;
        std::uint32_t pages      = 0;
        std::uint32_t generation = 1;
        bool active              = false;
        bool raw                  = false;
    };

    struct FreeExtent {
        std::size_t offset = 0;
        std::size_t bytes  = 0;
    };

    [[nodiscard]] std::optional<std::uint32_t> find_layout(const HostKVPageLayout& layout)
        const noexcept;
    [[nodiscard]] std::optional<std::size_t> find_free_extent(std::size_t bytes) const noexcept;
    [[nodiscard]] std::size_t plan_bytes(const HostKVAllocationPlan& entry) const noexcept;
    [[nodiscard]] bool view_matches(const HostKVAllocationView& view) const noexcept;
    [[nodiscard]] std::uint32_t take_descriptor() noexcept;
    [[nodiscard]] bool release_descriptor(std::uint32_t descriptor, std::uint32_t generation)
        noexcept;
    void insert_free_extent(FreeExtent extent) noexcept;
    [[nodiscard]] std::byte* allocation_data(const Descriptor& descriptor) const noexcept;
    void bump_revision() noexcept;

    std::optional<PinnedHostBuffer> backing_;
    std::size_t capacity_bytes_ = 0;
    std::size_t occupied_bytes_ = 0;
    std::vector<HostKVPageLayout> layouts_;
    std::vector<Descriptor> descriptors_;
    std::vector<std::uint32_t> free_descriptors_;
    std::vector<FreeExtent> free_extents_;
    std::uint64_t revision_ = 1;
};

} // namespace ninfer