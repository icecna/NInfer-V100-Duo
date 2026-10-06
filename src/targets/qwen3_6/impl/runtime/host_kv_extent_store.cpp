#include "targets/qwen3_6/impl/runtime/host_kv_extent_store.h"

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6::detail {

HostKvExtentStore::HostKvExtentStore(HostKVArena& arena, KvPageRegistry& registry,
                                     std::uint32_t descriptor_capacity)
    : arena_(&arena), registry_(&registry) {
    if (descriptor_capacity == 0) {
        throw std::invalid_argument("host extent descriptor capacity is zero");
    }
    extents_.resize(descriptor_capacity);
}

HostKvExtentStore::Reservation::~Reservation() {
    if (owner_ != nullptr) { owner_->abort(index_, generation_); }
}

HostKvExtentStore::Reservation::Reservation(Reservation&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_),
      generation_(other.generation_) {}

HostKvExtentStore::Reservation&
HostKvExtentStore::Reservation::operator=(Reservation&& other) noexcept {
    if (this == &other) { return *this; }
    if (owner_ != nullptr) { owner_->abort(index_, generation_); }
    owner_      = std::exchange(other.owner_, nullptr);
    index_      = other.index_;
    generation_ = other.generation_;
    return *this;
}

HostKvExtentStore::Capability::~Capability() {
    if (owner_ != nullptr) { owner_->release(index_, generation_); }
}

HostKvExtentStore::Capability::Capability(Capability&& other) noexcept
    : owner_(std::exchange(other.owner_, nullptr)), index_(other.index_),
      generation_(other.generation_) {}

HostKvExtentStore::Capability&
HostKvExtentStore::Capability::operator=(Capability&& other) noexcept {
    if (this == &other) { return *this; }
    if (owner_ != nullptr) { owner_->release(index_, generation_); }
    owner_      = std::exchange(other.owner_, nullptr);
    index_      = other.index_;
    generation_ = other.generation_;
    return *this;
}

HostKvExtentStore::Extent& HostKvExtentStore::require(std::uint32_t index,
                                                      std::uint32_t generation) {
    if (index >= extents_.size()) { throw std::out_of_range("host extent index out of range"); }
    Extent& extent = extents_[index];
    if (extent.state == State::Free || extent.generation != generation) {
        throw std::logic_error("host extent handle is stale");
    }
    return extent;
}

std::optional<HostKvExtentStore::Reservation>
HostKvExtentStore::reserve(const HostKVPageLayout& layout, std::span<const std::uint32_t> pages) {
    if (pages.empty() || pages.size() > std::numeric_limits<std::uint32_t>::max()) {
        return std::nullopt;
    }
    for (const std::uint32_t page : pages) {
        if (!registry_->can_pin_source(page)) { return std::nullopt; }
    }

    std::optional<HostKVAllocation> allocation =
        arena_->allocate(layout, static_cast<std::uint32_t>(pages.size()));
    if (!allocation) { return std::nullopt; }

    std::uint32_t index = 0;
    bool found          = false;
    for (std::uint32_t slot = 0; slot < extents_.size(); ++slot) {
        if (extents_[slot].state == State::Free) {
            index = slot;
            found = true;
            break;
        }
    }
    if (!found) { return std::nullopt; }

    const std::size_t bytes =
        allocation->raw_bytes() != 0
            ? allocation->raw_bytes()
            : static_cast<std::size_t>(allocation->page_count()) * layout.page_stride;
    Extent& extent    = extents_[index];
    extent.allocation = std::move(*allocation);
    extent.pages.assign(pages.begin(), pages.end());
    extent.bytes      = bytes;
    extent.generation = extent.generation + 1U;
    extent.state      = State::Reserved;
    for (const std::uint32_t page : extent.pages) { registry_->pin_source(page); }
    return Reservation(*this, index, extent.generation);
}

std::optional<HostKvExtentStore::Capability>
HostKvExtentStore::publish(Reservation&& reservation) {
    if (!reservation.valid()) { return std::nullopt; }
    Extent& extent     = require(reservation.index_, reservation.generation_);
    reservation.owner_ = nullptr;
    for (const std::uint32_t page : extent.pages) {
        registry_->attach_host_replica(page);
        registry_->unpin_source(page);
    }
    extent.state = State::Published;
    ++occupied_;
    occupied_bytes_ += extent.bytes;
    return Capability(*this, reservation.index_, reservation.generation_);
}

void HostKvExtentStore::abort(Reservation&& reservation) noexcept {
    if (!reservation.valid()) { return; }
    const std::uint32_t index      = reservation.index_;
    const std::uint32_t generation = reservation.generation_;
    reservation.owner_             = nullptr;
    abort(index, generation);
}

void HostKvExtentStore::release(Capability&& capability) noexcept {
    if (!capability.valid()) { return; }
    const std::uint32_t index      = capability.index_;
    const std::uint32_t generation = capability.generation_;
    capability.owner_              = nullptr;
    release(index, generation);
}

void HostKvExtentStore::abort(std::uint32_t index, std::uint32_t generation) noexcept {
    if (index >= extents_.size()) { return; }
    Extent& extent = extents_[index];
    if (extent.state != State::Reserved || extent.generation != generation) { return; }
    for (const std::uint32_t page : extent.pages) { registry_->unpin_source(page); }
    extent.allocation = HostKVAllocation{};
    extent.pages.clear();
    extent.bytes = 0;
    extent.state = State::Free;
    extent.generation = extent.generation + 1U;
}

void HostKvExtentStore::release(std::uint32_t index, std::uint32_t generation) noexcept {
    if (index >= extents_.size()) { return; }
    Extent& extent = extents_[index];
    if (extent.state != State::Published || extent.generation != generation) { return; }
    for (const std::uint32_t page : extent.pages) { registry_->detach_host_replica(page); }
    occupied_bytes_ -= extent.bytes;
    --occupied_;
    extent.allocation = HostKVAllocation{};
    extent.pages.clear();
    extent.bytes = 0;
    extent.state = State::Free;
    extent.generation = extent.generation + 1U;
}

bool HostKvExtentStore::page_releasable(std::uint32_t page) const {
    return registry_->references(page) == 0 && registry_->source_pins(page) == 0;
}

void HostKvExtentStore::release_unreferenced() {
    for (std::uint32_t index = 0; index < extents_.size(); ++index) {
        Extent& extent = extents_[index];
        if (extent.state != State::Published) { continue; }
        const bool releasable =
            std::all_of(extent.pages.begin(), extent.pages.end(),
                        [this](std::uint32_t page) { return page_releasable(page); });
        if (releasable) { release(index, extent.generation); }
    }
}

HostKVAllocationView HostKvExtentStore::view(const Capability& capability) const {
    if (capability.index_ >= extents_.size()) {
        throw std::out_of_range("host extent index out of range");
    }
    return arena_->view(extents_[capability.index_].allocation);
}

} // namespace ninfer::targets::qwen3_6::detail
