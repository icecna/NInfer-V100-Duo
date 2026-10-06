#pragma once

// Host-side ownership registry for the logical KV pages shared by saved checkpoints and the
// sequences that reference them. The device pool owns physical storage; this registry owns the
// facts that decide whether a page may be pinned, shared, truncated, or recycled:
//
//   references           address spaces that list the page
//   active_references    address spaces currently decoding or prefilling against the page
//   writer_references    the single address space appending to the page
//   source_pins          in-flight copies that must not observe a mutation
//   protected_columns    columns a published checkpoint depends on
//   committed_columns    columns committed to the device page
//
// All operations are caller-serialized (the executor thread); there is no locking.

#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

class KvPageRegistry {
public:
    explicit KvPageRegistry(std::uint32_t page_capacity);

    [[nodiscard]] std::uint32_t capacity() const noexcept { return capacity_; }
    [[nodiscard]] std::uint32_t occupied() const noexcept {
        return capacity_ - static_cast<std::uint32_t>(free_.size());
    }

    // Take a free page owned by one address space: references=1. The owner sets a writer while it
    // appends to the page.
    [[nodiscard]] std::optional<std::uint32_t> materialize() noexcept;
    [[nodiscard]] bool occupied(std::uint32_t page) const;

    void retain_reference(std::uint32_t page);
    void release_reference(std::uint32_t page);
    [[nodiscard]] std::uint32_t references(std::uint32_t page) const;

    void retain_active_reference(std::uint32_t page);
    void release_active_reference(std::uint32_t page);
    [[nodiscard]] std::uint32_t active_references(std::uint32_t page) const;

    void set_writer(std::uint32_t page);
    void release_writer(std::uint32_t page);
    [[nodiscard]] std::uint32_t writer_references(std::uint32_t page) const;

    [[nodiscard]] bool can_pin_source(std::uint32_t page) const;
    void pin_source(std::uint32_t page);
    void unpin_source(std::uint32_t page);
    [[nodiscard]] std::uint32_t source_pins(std::uint32_t page) const;

    void protect_columns(std::uint32_t page, std::uint32_t columns);
    void clear_protection(std::uint32_t page);
    [[nodiscard]] std::uint32_t protected_columns(std::uint32_t page) const;

    void commit_columns(std::uint32_t page, std::uint32_t columns);
    [[nodiscard]] std::uint32_t committed_columns(std::uint32_t page) const;

    void attach_host_replica(std::uint32_t page);
    void detach_host_replica(std::uint32_t page);
    [[nodiscard]] bool host_resident(std::uint32_t page) const;

    [[nodiscard]] bool can_release_reference(std::uint32_t page) const;
    [[nodiscard]] bool can_dematerialize(std::uint32_t page) const;

private:
    struct Page {
        std::uint32_t generation        = 1;
        std::uint32_t references        = 0;
        std::uint32_t active_references = 0;
        std::uint32_t writer_references = 0;
        std::uint32_t source_pins       = 0;
        std::uint32_t protected_columns = 0;
        std::uint32_t committed_columns = 0;
        bool occupied                   = false;
        bool host_resident              = false;
    };

    [[nodiscard]] Page& at(std::uint32_t page);
    [[nodiscard]] const Page& at(std::uint32_t page) const;
    void recycle_if_unreferenced(std::uint32_t page);

    std::uint32_t capacity_ = 0;
    std::vector<Page> pages_;
    std::vector<std::uint32_t> free_;
};

// An ordered, front-to-back list of the pages one sequence or checkpoint owns, with the token
// frontier committed to device and the frontier a published checkpoint depends on.
class KvAddressSpace {
public:
    // Materialize `page_count` fresh pages owned by this address space.
    [[nodiscard]] static KvAddressSpace create(KvPageRegistry& registry, std::uint32_t page_count,
                                               std::uint32_t tokens_per_page = 64);
    // Share every page of `source`, adding one reference per page.
    [[nodiscard]] static KvAddressSpace fork(KvPageRegistry& registry,
                                             const KvAddressSpace& source);

    ~KvAddressSpace();
    KvAddressSpace(KvAddressSpace&&) noexcept;
    KvAddressSpace& operator=(KvAddressSpace&&) noexcept;
    KvAddressSpace(const KvAddressSpace&)            = delete;
    KvAddressSpace& operator=(const KvAddressSpace&) = delete;

    [[nodiscard]] std::span<const std::uint32_t> pages() const noexcept { return pages_; }
    [[nodiscard]] std::uint32_t committed_frontier() const noexcept { return committed_frontier_; }
    [[nodiscard]] std::uint32_t checkpoint_frontier() const noexcept { return checkpoint_frontier_; }

    // Commit `tokens` to the device pages, recording per-page committed columns.
    void commit_frontier(std::uint32_t tokens);
    // Record that a published checkpoint depends on `tokens`; protects the fully covered prefix.
    void set_checkpoint_requirement(std::uint32_t tokens);

    [[nodiscard]] bool can_truncate_inactive_prefix(std::uint32_t tokens) const;
    // Drop the suffix pages above `tokens`; requires can_truncate_inactive_prefix.
    void truncate_inactive_prefix(std::uint32_t tokens);

    [[nodiscard]] bool can_release() const;
    void release();

private:
    KvAddressSpace(KvPageRegistry& registry, std::vector<std::uint32_t> pages,
                   std::uint32_t tokens_per_page);

    [[nodiscard]] std::uint32_t pages_for_tokens(std::uint32_t tokens) const noexcept;

    KvPageRegistry* registry_         = nullptr;
    std::vector<std::uint32_t> pages_;
    std::uint32_t tokens_per_page_    = 64;
    std::uint32_t committed_frontier_ = 0;
    std::uint32_t checkpoint_frontier_ = 0;
    bool released_                    = false;
};

} // namespace ninfer::targets::qwen3_6::detail
