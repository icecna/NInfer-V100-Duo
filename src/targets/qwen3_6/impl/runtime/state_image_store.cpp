#include "targets/qwen3_6/impl/runtime/state_image_store.h"

#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6::detail {
namespace {

constexpr std::size_t kHostImageAlignment = 256;

std::size_t align_up(std::size_t value) noexcept {
    return (value + kHostImageAlignment - 1) / kHostImageAlignment * kHostImageAlignment;
}

} // namespace

std::size_t StateImageSpec::host_bytes() const noexcept {
    std::size_t bytes = linear_state_slot_bytes(linear);
    if (hidden > 0) { bytes += static_cast<std::size_t>(hidden) * sizeof(std::uint16_t); }
    bytes += dflash_local_bytes;
    return align_up(bytes);
}

StateImageStore::StateImageStore(StateImageSpec spec, std::uint32_t logical_capacity)
    : spec_(spec) {
    if (logical_capacity == 0) {
        throw std::invalid_argument("state image store capacity is zero");
    }
    if (spec_.hidden < 0) { throw std::invalid_argument("state image hidden size is negative"); }
    (void)linear_state_slot_bytes(spec_.linear); // validates the linear pool spec
    slots_.resize(logical_capacity);
    free_.reserve(logical_capacity);
    for (std::uint32_t slot = logical_capacity; slot > 0; --slot) { free_.push_back(slot - 1); }
    free_count_ = logical_capacity;
}

StateImageStore::Slot& StateImageStore::at(std::uint32_t slot) {
    if (slot >= slots_.size()) { throw std::out_of_range("state image slot out of range"); }
    return slots_[slot];
}

const StateImageStore::Slot& StateImageStore::at(std::uint32_t slot) const {
    if (slot >= slots_.size()) { throw std::out_of_range("state image slot out of range"); }
    return slots_[slot];
}

void StateImageStore::recycle(std::uint32_t slot) noexcept {
    Slot& entry     = slots_[slot];
    entry.role      = StateImageRole::Free;
    entry.checkpoint_references = 0;
    entry.pins      = 0;
    entry.generation = entry.generation + 1U;
    free_.push_back(slot);
    ++free_count_;
}

std::uint32_t StateImageStore::checkpoint_count() const noexcept {
    std::uint32_t count = 0;
    for (const Slot& slot : slots_) {
        if (slot.role == StateImageRole::CheckpointImmutable) { ++count; }
    }
    return count;
}

StateImageRole StateImageStore::role(std::uint32_t slot) const { return at(slot).role; }

std::uint32_t StateImageStore::checkpoint_references(std::uint32_t slot) const {
    return at(slot).checkpoint_references;
}

std::uint32_t StateImageStore::pins(std::uint32_t slot) const { return at(slot).pins; }

std::optional<std::uint32_t> StateImageStore::reserve_destination() {
    if (free_.empty()) { return std::nullopt; }
    const std::uint32_t slot = free_.back();
    free_.pop_back();
    --free_count_;
    Slot& entry  = slots_[slot];
    entry.role   = StateImageRole::ReservedDestination;
    entry.generation = entry.generation + 1U;
    return slot;
}

bool StateImageStore::can_activate(std::uint32_t slot) const {
    const Slot& entry = at(slot);
    if (entry.role == StateImageRole::ReservedDestination) { return true; }
    return entry.role == StateImageRole::CheckpointImmutable &&
           entry.checkpoint_references == 0 && entry.pins == 0;
}

void StateImageStore::activate(std::uint32_t slot) {
    if (!can_activate(slot)) {
        throw std::logic_error("state image slot cannot be activated while it is shared or pinned");
    }
    at(slot).role = StateImageRole::ActiveMutable;
}

void StateImageStore::freeze(std::uint32_t slot) {
    Slot& entry = at(slot);
    if (entry.role != StateImageRole::ActiveMutable) {
        throw std::logic_error("only an active state image can be frozen");
    }
    entry.role                  = StateImageRole::CheckpointImmutable;
    entry.checkpoint_references = 1;
}

void StateImageStore::deactivate(std::uint32_t slot) {
    Slot& entry = at(slot);
    if (entry.role != StateImageRole::ActiveMutable || entry.checkpoint_references != 0 ||
        entry.pins != 0) {
        throw std::logic_error("state image slot is still referenced or pinned");
    }
    recycle(slot);
}

void StateImageStore::retain_checkpoint_reference(std::uint32_t slot) {
    Slot& entry = at(slot);
    if (entry.role != StateImageRole::CheckpointImmutable) {
        throw std::logic_error("only a frozen state image can gain a checkpoint reference");
    }
    ++entry.checkpoint_references;
}

void StateImageStore::release_checkpoint_reference(std::uint32_t slot) {
    Slot& entry = at(slot);
    if (entry.role != StateImageRole::CheckpointImmutable ||
        entry.checkpoint_references == 0) {
        throw std::logic_error("state image checkpoint reference underflow");
    }
    --entry.checkpoint_references;
    if (entry.checkpoint_references == 0 && entry.pins == 0) { recycle(slot); }
}

bool StateImageStore::can_release_checkpoint_owner(std::uint32_t slot) const {
    const Slot& entry = at(slot);
    return entry.role == StateImageRole::CheckpointImmutable &&
           entry.checkpoint_references > 0 && entry.pins == 0;
}

void StateImageStore::release_checkpoint_owner(std::uint32_t slot) {
    if (!can_release_checkpoint_owner(slot)) {
        throw std::logic_error("state image checkpoint is still pinned");
    }
    recycle(slot);
}

void StateImageStore::pin(std::uint32_t slot) {
    Slot& entry = at(slot);
    if (entry.role == StateImageRole::Free) {
        throw std::logic_error("cannot pin a free state image slot");
    }
    ++entry.pins;
}

void StateImageStore::unpin(std::uint32_t slot) {
    Slot& entry = at(slot);
    if (entry.pins == 0) { throw std::logic_error("state image pin underflow"); }
    --entry.pins;
    if (entry.pins == 0 && entry.role == StateImageRole::CheckpointImmutable &&
        entry.checkpoint_references == 0) {
        recycle(slot);
    }
}

HostStatePool::HostStatePool(HostKVArena& arena, std::size_t image_bytes, std::uint32_t capacity)
    : arena_(&arena), image_bytes_(image_bytes) {
    if (image_bytes == 0 || capacity == 0) {
        throw std::invalid_argument("host state pool needs a positive image size and capacity");
    }
    images_.resize(capacity);
    free_.reserve(capacity);
    for (std::uint32_t image = capacity; image > 0; --image) { free_.push_back(image - 1); }
}

std::optional<std::uint32_t> HostStatePool::allocate() {
    if (free_.empty()) { return std::nullopt; }
    std::optional<HostKVAllocation> allocation = arena_->allocate_raw(image_bytes_);
    if (!allocation) { return std::nullopt; }
    const std::uint32_t image = free_.back();
    free_.pop_back();
    images_[image] = std::move(*allocation);
    return image;
}

void HostStatePool::release(std::uint32_t image) noexcept {
    if (image >= images_.size() || !images_[image].has_value()) { return; }
    images_[image].reset();
    free_.push_back(image);
}

std::byte* HostStatePool::data(std::uint32_t image) {
    if (image >= images_.size() || !images_[image].has_value()) {
        throw std::out_of_range("host state image is not allocated");
    }
    return arena_->raw_view(*images_[image]).data();
}

} // namespace ninfer::targets::qwen3_6::detail
