#include "core/host_kv_arena.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer {
namespace {

constexpr std::size_t kHostKVAlignment = 256;

std::size_t checked_add(std::size_t a, std::size_t b, const char* label) {
    if (b > std::numeric_limits<std::size_t>::max() - a) { throw std::overflow_error(label); }
    return a + b;
}

std::size_t checked_mul(std::size_t a, std::size_t b, const char* label) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::overflow_error(label);
    }
    return a * b;
}

std::size_t align_up(std::size_t value, std::size_t alignment, const char* label) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument(std::string(label) + " alignment must be a power of two");
    }
    const std::size_t mask = alignment - 1;
    if (value > std::numeric_limits<std::size_t>::max() - mask) {
        throw std::overflow_error(std::string(label) + " alignment overflow");
    }
    return (value + mask) & ~mask;
}

void increment_generation(std::uint32_t& generation) noexcept {
    ++generation;
    if (generation == 0) { ++generation; }
}

} // namespace

HostKVPageLayout plan_host_kv_page_layout(const PagedKVPoolSpec& spec) {
    if (spec.planes.empty()) { throw std::invalid_argument("Host KV page layout requires planes"); }

    HostKVPageLayout out;
    out.device_plane_order = spec.plane_order;
    out.planes.reserve(spec.planes.size());
    std::size_t cursor = 0;
    for (const PagedKVPlaneSpec& plane : spec.planes) {
        if (plane.leading_extent <= 0 || plane.head_extent <= 0) {
            throw std::invalid_argument("Host KV plane extents must be positive");
        }
        cursor = align_up(cursor, kHostKVAlignment, "Host KV plane");
        const std::size_t head_bytes = checked_mul(
            checked_mul(static_cast<std::size_t>(plane.leading_extent),
                        static_cast<std::size_t>(kPagedKVPageSize),
                        "Host KV head payload overflow"),
            dtype_size(plane.dtype), "Host KV head payload overflow");
        const std::size_t page_bytes =
            checked_mul(head_bytes, static_cast<std::size_t>(plane.head_extent),
                        "Host KV plane payload overflow");
        out.planes.push_back(HostKVPlaneLayout{
            .dtype                = plane.dtype,
            .leading_extent       = plane.leading_extent,
            .head_extent          = plane.head_extent,
            .offset             = cursor,
            .page_payload_bytes = page_bytes,
            .head_payload_bytes = head_bytes,
        });
        cursor = checked_add(cursor, page_bytes, "Host KV page payload overflow");
    }
    out.page_stride = align_up(cursor, kHostKVAlignment, "Host KV page record");
    return out;
}

bool HostKVAllocationView::valid() const noexcept {
    return arena_ != nullptr && data_ != nullptr && arena_->view_matches(*this);
}

const HostKVPageLayout& HostKVAllocationView::layout() const {
    if (!valid() || layout_ == nullptr) { throw std::logic_error("Host KV view is stale"); }
    return *layout_;
}

HostKVAllocationView HostKVAllocationView::subview(std::uint32_t begin, std::uint32_t count)
    const {
    if (!valid() || count == 0 || begin > page_count_ || count > page_count_ - begin) {
        throw std::out_of_range("Host KV subview is outside its allocation");
    }
    return HostKVAllocationView(
        arena_, revision_, data_ + static_cast<std::size_t>(begin) * layout_->page_stride,
        layout_, count);
}

HostKVAllocation::~HostKVAllocation() { (void)release(); }

HostKVAllocation::HostKVAllocation(HostKVAllocation&& other) noexcept
    : arena_(other.arena_), descriptor_(other.descriptor_), generation_(other.generation_) {
    other.disarm();
}

HostKVAllocation& HostKVAllocation::operator=(HostKVAllocation&& other) noexcept {
    if (this == &other) { return *this; }
    (void)release();
    arena_      = other.arena_;
    descriptor_ = other.descriptor_;
    generation_ = other.generation_;
    other.disarm();
    return *this;
}

std::uint32_t HostKVAllocation::page_count() const noexcept {
    if (!arena_ || descriptor_ >= arena_->descriptors_.size()) { return 0; }
    const HostKVArena::Descriptor& descriptor = arena_->descriptors_[descriptor_];
    return descriptor.active && descriptor.generation == generation_ ? descriptor.pages : 0;
}

bool HostKVAllocation::release() noexcept {
    if (!arena_) { return false; }
    const bool released = arena_->release_descriptor(descriptor_, generation_);
    disarm();
    return released;
}

void HostKVAllocation::disarm() noexcept {
    arena_      = nullptr;
    descriptor_ = 0;
    generation_ = 0;
}

HostKVArena::HostKVArena(std::size_t capacity_bytes,
                         std::span<const HostKVPageLayout> supported_layouts)
    : capacity_bytes_(capacity_bytes),
      layouts_(supported_layouts.begin(), supported_layouts.end()) {
    for (std::size_t index = 0; index < layouts_.size(); ++index) {
        const HostKVPageLayout planned =
            plan_host_kv_page_layout(PagedKVPoolSpec{
                .plane_order = layouts_[index].device_plane_order,
                .planes      = [&] {
                    std::vector<PagedKVPlaneSpec> specs;
                    specs.reserve(layouts_[index].planes.size());
                    for (const HostKVPlaneLayout& plane : layouts_[index].planes) {
                        specs.push_back({plane.dtype, plane.leading_extent, plane.head_extent});
                    }
                    return specs;
                }(),
            });
        if (planned != layouts_[index]) {
            throw std::invalid_argument("Host KV arena received an inconsistent page layout");
        }
        for (std::size_t previous = 0; previous < index; ++previous) {
            if (layouts_[previous] == layouts_[index]) {
                throw std::invalid_argument("Host KV arena contains a duplicate page layout");
            }
        }
    }
    if (capacity_bytes_ == 0) { return; }
    if (layouts_.empty()) {
        throw std::invalid_argument("Non-empty Host KV arena requires supported page layouts");
    }

    backing_.emplace(capacity_bytes_);
    const auto smallest = std::min_element(
        layouts_.begin(), layouts_.end(), [](const HostKVPageLayout& a,
                                             const HostKVPageLayout& b) {
            return a.page_stride < b.page_stride;
        });
    const std::size_t maximum_descriptors = capacity_bytes_ / smallest->page_stride;
    if (maximum_descriptors == 0 || maximum_descriptors > std::numeric_limits<std::uint32_t>::max()) {
        throw std::overflow_error("Host KV arena descriptor capacity exceeds uint32");
    }
    descriptors_.resize(maximum_descriptors);
    free_descriptors_.reserve(maximum_descriptors);
    for (std::size_t index = maximum_descriptors; index > 0; --index) {
        free_descriptors_.push_back(static_cast<std::uint32_t>(index - 1));
    }
    free_extents_.reserve(maximum_descriptors + 1);
    free_extents_.push_back({0, capacity_bytes_});
}

const HostKVPageLayout* HostKVArena::layout_for(const PagedKVPoolSpec& spec) const noexcept {
    const HostKVPageLayout planned = plan_host_kv_page_layout(spec);
    const auto it = std::find(layouts_.begin(), layouts_.end(), planned);
    return it == layouts_.end() ? nullptr : &*it;
}

std::optional<std::size_t> HostKVArena::find_free_extent(std::size_t bytes) const noexcept {
    for (std::size_t index = 0; index < free_extents_.size(); ++index) {
        if (free_extents_[index].bytes >= bytes) { return index; }
    }
    return std::nullopt;
}

bool HostKVArena::can_allocate(const HostKVPageLayout& layout, std::uint32_t pages) const
    noexcept {
    if (pages == 0 || free_descriptors_.empty() || !find_layout(layout) ||
        layout.page_stride > std::numeric_limits<std::size_t>::max() / pages) {
        return false;
    }
    return find_free_extent(layout.page_stride * static_cast<std::size_t>(pages)).has_value();
}

bool HostKVArena::can_allocate_all(std::span<const HostKVAllocationPlan> plan) const noexcept {
    if (plan.size() > free_descriptors_.size()) { return false; }
    // Simulate the sequential allocation on a scratch copy of the free extents: each entry
    // takes the first extent large enough and consumes its front, exactly as allocate() does.
    auto extents = free_extents_;
    for (const HostKVAllocationPlan& entry : plan) {
        if (entry.layout == nullptr || entry.pages == 0 || !find_layout(*entry.layout) ||
            entry.layout->page_stride >
                std::numeric_limits<std::size_t>::max() / entry.pages) {
            return false;
        }
        const std::size_t bytes =
            entry.layout->page_stride * static_cast<std::size_t>(entry.pages);
        bool placed = false;
        for (std::size_t index = 0; index < extents.size(); ++index) {
            if (extents[index].bytes < bytes) { continue; }
            extents[index].offset += bytes;
            extents[index].bytes -= bytes;
            if (extents[index].bytes == 0) { extents.erase(extents.begin() + index); }
            placed = true;
            break;
        }
        if (!placed) { return false; }
    }
    return true;
}

std::optional<HostKVAllocation> HostKVArena::allocate(const HostKVPageLayout& layout,
                                                      std::uint32_t pages) noexcept {
    const std::optional<std::uint32_t> layout_index = find_layout(layout);
    if (!layout_index || pages == 0 || free_descriptors_.empty() ||
        layout.page_stride > std::numeric_limits<std::size_t>::max() / pages) {
        return std::nullopt;
    }
    const std::size_t bytes = layout.page_stride * static_cast<std::size_t>(pages);
    const std::optional<std::size_t> free_index = find_free_extent(bytes);
    if (!free_index) { return std::nullopt; }

    const std::uint32_t descriptor_index = take_descriptor();
    if (descriptor_index == std::numeric_limits<std::uint32_t>::max()) { return std::nullopt; }
    FreeExtent& free         = free_extents_[*free_index];
    const std::size_t offset = free.offset;
    free.offset += bytes;
    free.bytes -= bytes;
    if (free.bytes == 0) {
        free_extents_.erase(free_extents_.begin() + static_cast<std::ptrdiff_t>(*free_index));
    }

    Descriptor& descriptor = descriptors_[descriptor_index];
    descriptor.offset      = offset;
    descriptor.bytes       = bytes;
    descriptor.layout      = *layout_index;
    descriptor.pages       = pages;
    descriptor.active      = true;
    occupied_bytes_ += bytes;
    bump_revision();
    return HostKVAllocation(*this, descriptor_index, descriptor.generation);
}

HostKVAllocationView HostKVArena::view(const HostKVAllocation& allocation) const {
    if (!allocation.valid()) {
        throw std::invalid_argument("Cannot view a stale Host KV allocation");
    }
    const Descriptor& descriptor = descriptors_[allocation.descriptor_];
    return HostKVAllocationView(this, revision_, allocation_data(descriptor),
                                &layouts_[descriptor.layout], descriptor.pages);
}

bool HostKVArena::view_matches(const HostKVAllocationView& view) const noexcept {
    return view.revision_ == revision_;
}

std::optional<std::uint32_t> HostKVArena::find_layout(const HostKVPageLayout& layout) const
    noexcept {
    const auto it = std::find(layouts_.begin(), layouts_.end(), layout);
    if (it == layouts_.end()) { return std::nullopt; }
    return static_cast<std::uint32_t>(it - layouts_.begin());
}

std::uint32_t HostKVArena::take_descriptor() noexcept {
    if (free_descriptors_.empty()) { return std::numeric_limits<std::uint32_t>::max(); }
    const std::uint32_t out = free_descriptors_.back();
    free_descriptors_.pop_back();
    return out;
}

bool HostKVArena::release_descriptor(std::uint32_t descriptor_index,
                                     std::uint32_t generation) noexcept {
    if (descriptor_index >= descriptors_.size()) { return false; }
    Descriptor& descriptor = descriptors_[descriptor_index];
    if (!descriptor.active || descriptor.generation != generation) { return false; }

    const FreeExtent released{descriptor.offset, descriptor.bytes};
    occupied_bytes_ -= descriptor.bytes;
    descriptor.active = false;
    descriptor.offset = 0;
    descriptor.bytes  = 0;
    descriptor.pages  = 0;
    increment_generation(descriptor.generation);
    free_descriptors_.push_back(descriptor_index);
    insert_free_extent(released);
    bump_revision();
    return true;
}

void HostKVArena::insert_free_extent(FreeExtent extent) noexcept {
    const auto position = std::lower_bound(
        free_extents_.begin(), free_extents_.end(), extent.offset,
        [](const FreeExtent& candidate, std::size_t offset) { return candidate.offset < offset; });
    auto inserted = free_extents_.insert(position, extent);
    if (inserted != free_extents_.begin()) {
        auto previous = inserted - 1;
        if (previous->offset + previous->bytes == inserted->offset) {
            previous->bytes += inserted->bytes;
            inserted = free_extents_.erase(inserted);
            inserted = previous;
        }
    }
    auto next = inserted + 1;
    if (next != free_extents_.end() && inserted->offset + inserted->bytes == next->offset) {
        inserted->bytes += next->bytes;
        free_extents_.erase(next);
    }
}

std::byte* HostKVArena::allocation_data(const Descriptor& descriptor) const noexcept {
    if (!backing_) { return nullptr; }
    return static_cast<std::byte*>(backing_->data()) + descriptor.offset;
}

void HostKVArena::bump_revision() noexcept {
    ++revision_;
    if (revision_ == 0) { ++revision_; }
}

} // namespace ninfer