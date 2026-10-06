#pragma once

// Host-side registry for checkpoint state images: the GDN (linear attention) conv/recurrent
// pools, the continuation hidden, and the optional DFlash local state that a reusable checkpoint
// must snapshot alongside its KV. The device pool owns the physical slots; this registry owns the
// logical role, checkpoint references, and transfer pins that decide whether a slot may be
// activated, frozen, shared, or recycled.
//
// All operations are caller-serialized (the executor thread); there is no locking.

#include "core/linear_attention_state.h"
#include "core/host_kv_arena.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

struct StateImageSpec {
    LinearAttentionStatePoolSpec linear{};
    std::int32_t hidden           = 0;
    // Byte count of the speculative backend's private local state (for DFlash, its cyclic KV);
    // zero keeps the backend stateless across a checkpoint.
    std::size_t dflash_local_bytes = 0;

    [[nodiscard]] std::size_t host_bytes() const noexcept;
};

enum class StateImageRole : std::uint8_t {
    Free,
    ActiveMutable,
    CheckpointImmutable,
    ReservedDestination,
};

class StateImageStore {
public:
    StateImageStore(StateImageSpec spec, std::uint32_t logical_capacity);

    [[nodiscard]] const StateImageSpec& spec() const noexcept { return spec_; }
    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(slots_.size());
    }
    [[nodiscard]] std::uint32_t occupied() const noexcept { return capacity() - free_count_; }
    [[nodiscard]] std::uint32_t checkpoint_count() const noexcept;

    [[nodiscard]] StateImageRole role(std::uint32_t slot) const;
    [[nodiscard]] std::uint32_t checkpoint_references(std::uint32_t slot) const;
    [[nodiscard]] std::uint32_t pins(std::uint32_t slot) const;

    // Allocation.
    [[nodiscard]] std::optional<std::uint32_t> reserve_destination();
    [[nodiscard]] bool can_activate(std::uint32_t slot) const;
    void activate(std::uint32_t slot);
    // ActiveMutable -> CheckpointImmutable with one owning checkpoint reference.
    void freeze(std::uint32_t slot);
    // ActiveMutable -> Free; requires no references or pins.
    void deactivate(std::uint32_t slot);

    void retain_checkpoint_reference(std::uint32_t slot);
    void release_checkpoint_reference(std::uint32_t slot);
    [[nodiscard]] bool can_release_checkpoint_owner(std::uint32_t slot) const;
    void release_checkpoint_owner(std::uint32_t slot);

    void pin(std::uint32_t slot);
    void unpin(std::uint32_t slot);

private:
    struct Slot {
        std::uint32_t generation          = 1;
        StateImageRole role               = StateImageRole::Free;
        std::uint32_t checkpoint_references = 0;
        std::uint32_t pins                = 0;
    };

    [[nodiscard]] Slot& at(std::uint32_t slot);
    [[nodiscard]] const Slot& at(std::uint32_t slot) const;
    void recycle(std::uint32_t slot) noexcept;

    StateImageSpec spec_;
    std::vector<Slot> slots_;
    std::vector<std::uint32_t> free_;
    std::uint32_t free_count_ = 0;
};

// Fixed number of raw pinned host images of `image_bytes` each, over a shared HostKVArena.
class HostStatePool {
public:
    HostStatePool(HostKVArena& arena, std::size_t image_bytes, std::uint32_t capacity);

    [[nodiscard]] std::uint32_t capacity() const noexcept {
        return static_cast<std::uint32_t>(images_.size());
    }
    [[nodiscard]] std::uint32_t occupied() const noexcept {
        return capacity() - static_cast<std::uint32_t>(free_.size());
    }
    [[nodiscard]] std::size_t image_bytes() const noexcept { return image_bytes_; }
    [[nodiscard]] std::size_t occupied_bytes() const noexcept {
        return static_cast<std::size_t>(occupied()) * image_bytes_;
    }

    [[nodiscard]] std::optional<std::uint32_t> allocate();
    void release(std::uint32_t image) noexcept;
    [[nodiscard]] std::byte* data(std::uint32_t image);

private:
    HostKVArena* arena_       = nullptr;
    std::size_t image_bytes_  = 0;
    std::vector<std::optional<HostKVAllocation>> images_;
    std::vector<std::uint32_t> free_;
};

} // namespace ninfer::targets::qwen3_6::detail
