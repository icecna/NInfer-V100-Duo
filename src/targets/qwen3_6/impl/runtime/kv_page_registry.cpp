#include "targets/qwen3_6/impl/runtime/kv_page_registry.h"

#include <algorithm>
#include <stdexcept>
#include <utility>

namespace ninfer::targets::qwen3_6::detail {

KvPageRegistry::KvPageRegistry(std::uint32_t page_capacity) : capacity_(page_capacity) {
    if (page_capacity == 0) { throw std::invalid_argument("KV page registry capacity is zero"); }
    pages_.resize(page_capacity);
    free_.reserve(page_capacity);
    for (std::uint32_t page = page_capacity; page > 0; --page) { free_.push_back(page - 1); }
}

KvPageRegistry::Page& KvPageRegistry::at(std::uint32_t page) {
    if (page >= capacity_) { throw std::out_of_range("KV page index exceeds registry capacity"); }
    return pages_[page];
}

const KvPageRegistry::Page& KvPageRegistry::at(std::uint32_t page) const {
    if (page >= capacity_) { throw std::out_of_range("KV page index exceeds registry capacity"); }
    return pages_[page];
}

std::optional<std::uint32_t> KvPageRegistry::materialize() noexcept {
    if (free_.empty()) { return std::nullopt; }
    const std::uint32_t page = free_.back();
    free_.pop_back();
    Page& entry             = pages_[page];
    entry.generation        = entry.generation + 1U;
    entry.occupied          = true;
    entry.references        = 1;
    return page;
}

bool KvPageRegistry::occupied(std::uint32_t page) const { return at(page).occupied; }

void KvPageRegistry::retain_reference(std::uint32_t page) {
    Page& entry = at(page);
    if (!entry.occupied) { throw std::logic_error("cannot reference an unmaterialized KV page"); }
    ++entry.references;
}

void KvPageRegistry::release_reference(std::uint32_t page) {
    Page& entry = at(page);
    if (!entry.occupied || entry.references == 0) {
        throw std::logic_error("KV page reference underflow");
    }
    --entry.references;
    recycle_if_unreferenced(page);
}

std::uint32_t KvPageRegistry::references(std::uint32_t page) const { return at(page).references; }

void KvPageRegistry::retain_active_reference(std::uint32_t page) {
    Page& entry = at(page);
    if (!entry.occupied || entry.active_references >= entry.references) {
        throw std::logic_error("active KV page reference exceeds the address references");
    }
    ++entry.active_references;
}

void KvPageRegistry::release_active_reference(std::uint32_t page) {
    Page& entry = at(page);
    if (entry.active_references == 0) {
        throw std::logic_error("active KV page reference underflow");
    }
    --entry.active_references;
}

std::uint32_t KvPageRegistry::active_references(std::uint32_t page) const {
    return at(page).active_references;
}

void KvPageRegistry::set_writer(std::uint32_t page) {
    Page& entry = at(page);
    if (!entry.occupied || entry.writer_references != 0) {
        throw std::logic_error("KV page already has a writer");
    }
    entry.writer_references = 1;
}

void KvPageRegistry::release_writer(std::uint32_t page) {
    Page& entry = at(page);
    if (entry.writer_references == 0) { throw std::logic_error("KV page writer underflow"); }
    entry.writer_references = 0;
    recycle_if_unreferenced(page);
}

std::uint32_t KvPageRegistry::writer_references(std::uint32_t page) const {
    return at(page).writer_references;
}

bool KvPageRegistry::can_pin_source(std::uint32_t page) const {
    const Page& entry = at(page);
    return entry.occupied && entry.writer_references == 0;
}

void KvPageRegistry::pin_source(std::uint32_t page) {
    Page& entry = at(page);
    if (!can_pin_source(page)) {
        throw std::logic_error("KV page cannot be pinned while a writer holds it");
    }
    ++entry.source_pins;
}

void KvPageRegistry::unpin_source(std::uint32_t page) {
    Page& entry = at(page);
    if (entry.source_pins == 0) { throw std::logic_error("KV page source pin underflow"); }
    --entry.source_pins;
    recycle_if_unreferenced(page);
}

std::uint32_t KvPageRegistry::source_pins(std::uint32_t page) const {
    return at(page).source_pins;
}

void KvPageRegistry::protect_columns(std::uint32_t page, std::uint32_t columns) {
    Page& entry = at(page);
    entry.protected_columns = std::max(entry.protected_columns, columns);
}

void KvPageRegistry::clear_protection(std::uint32_t page) {
    at(page).protected_columns = 0;
}

std::uint32_t KvPageRegistry::protected_columns(std::uint32_t page) const {
    return at(page).protected_columns;
}

void KvPageRegistry::commit_columns(std::uint32_t page, std::uint32_t columns) {
    Page& entry = at(page);
    if (columns < entry.committed_columns) {
        throw std::logic_error("committed KV columns must not move backwards");
    }
    entry.committed_columns = columns;
}

std::uint32_t KvPageRegistry::committed_columns(std::uint32_t page) const {
    return at(page).committed_columns;
}

void KvPageRegistry::attach_host_replica(std::uint32_t page) { at(page).host_resident = true; }

void KvPageRegistry::detach_host_replica(std::uint32_t page) {
    Page& entry = at(page);
    entry.host_resident = false;
    recycle_if_unreferenced(page);
}

bool KvPageRegistry::host_resident(std::uint32_t page) const { return at(page).host_resident; }

bool KvPageRegistry::can_release_reference(std::uint32_t page) const {
    const Page& entry = at(page);
    return entry.occupied && entry.references > 0 && entry.active_references == 0 &&
           entry.source_pins == 0 && entry.writer_references == 0;
}

bool KvPageRegistry::can_dematerialize(std::uint32_t page) const {
    const Page& entry = at(page);
    return entry.occupied && entry.references == 0 && entry.active_references == 0 &&
           entry.source_pins == 0;
}

void KvPageRegistry::recycle_if_unreferenced(std::uint32_t page) {
    Page& entry = pages_[page];
    if (!entry.occupied || entry.references != 0 || entry.active_references != 0 ||
        entry.source_pins != 0 || entry.writer_references != 0 || entry.host_resident) {
        return;
    }
    entry     = Page{};
    free_.push_back(page);
}

KvAddressSpace::KvAddressSpace(KvPageRegistry& registry, std::vector<std::uint32_t> pages,
                               std::uint32_t tokens_per_page)
    : registry_(&registry), pages_(std::move(pages)), tokens_per_page_(tokens_per_page) {
    if (tokens_per_page == 0) { throw std::invalid_argument("KV page column count is zero"); }
}

KvAddressSpace KvAddressSpace::create(KvPageRegistry& registry, std::uint32_t page_count,
                                      std::uint32_t tokens_per_page) {
    std::vector<std::uint32_t> pages;
    pages.reserve(page_count);
    for (std::uint32_t index = 0; index < page_count; ++index) {
        const std::optional<std::uint32_t> page = registry.materialize();
        if (!page) {
            for (const std::uint32_t materialized : pages) {
                registry.release_reference(materialized);
            }
            throw std::length_error("KV page registry capacity exhausted");
        }
        pages.push_back(*page);
    }
    return KvAddressSpace(registry, std::move(pages), tokens_per_page);
}

KvAddressSpace KvAddressSpace::fork(KvPageRegistry& registry, const KvAddressSpace& source) {
    std::vector<std::uint32_t> pages = source.pages_;
    for (const std::uint32_t page : pages) { registry.retain_reference(page); }
    KvAddressSpace forked(registry, std::move(pages), source.tokens_per_page_);
    forked.committed_frontier_  = source.committed_frontier_;
    forked.checkpoint_frontier_ = source.checkpoint_frontier_;
    return forked;
}

KvAddressSpace::~KvAddressSpace() {
    if (registry_ != nullptr && !released_) {
        for (const std::uint32_t page : pages_) { registry_->release_reference(page); }
    }
}

KvAddressSpace::KvAddressSpace(KvAddressSpace&& other) noexcept
    : registry_(std::exchange(other.registry_, nullptr)), pages_(std::move(other.pages_)),
      tokens_per_page_(other.tokens_per_page_),
      committed_frontier_(other.committed_frontier_),
      checkpoint_frontier_(other.checkpoint_frontier_), released_(other.released_) {}

KvAddressSpace& KvAddressSpace::operator=(KvAddressSpace&& other) noexcept {
    if (this == &other) { return *this; }
    if (registry_ != nullptr && !released_) {
        for (const std::uint32_t page : pages_) { registry_->release_reference(page); }
    }
    registry_            = std::exchange(other.registry_, nullptr);
    pages_               = std::move(other.pages_);
    tokens_per_page_     = other.tokens_per_page_;
    committed_frontier_  = other.committed_frontier_;
    checkpoint_frontier_ = other.checkpoint_frontier_;
    released_            = other.released_;
    return *this;
}

std::uint32_t KvAddressSpace::pages_for_tokens(std::uint32_t tokens) const noexcept {
    return (tokens + tokens_per_page_ - 1) / tokens_per_page_;
}

void KvAddressSpace::commit_frontier(std::uint32_t tokens) {
    if (released_) { throw std::logic_error("KV address space is released"); }
    if (tokens < committed_frontier_) {
        throw std::logic_error("KV address space frontier must not move backwards");
    }
    const std::uint32_t covered = pages_for_tokens(tokens);
    for (std::uint32_t index = 0; index < covered && index < pages_.size(); ++index) {
        const std::uint32_t begin   = index * tokens_per_page_;
        const std::uint32_t columns = std::min(tokens_per_page_, tokens - begin);
        registry_->commit_columns(pages_[index], columns);
    }
    committed_frontier_ = tokens;
}

void KvAddressSpace::set_checkpoint_requirement(std::uint32_t tokens) {
    if (released_) { throw std::logic_error("KV address space is released"); }
    checkpoint_frontier_ = tokens;
    // A checkpoint depends on every page fully covered by its frontier; the partially covered last
    // page stays unprotected because its tail columns may still change.
    const std::uint32_t covered = tokens / tokens_per_page_;
    for (std::uint32_t index = 0; index < covered && index < pages_.size(); ++index) {
        registry_->protect_columns(pages_[index], tokens_per_page_);
    }
}

bool KvAddressSpace::can_truncate_inactive_prefix(std::uint32_t tokens) const {
    const std::uint32_t keep = pages_for_tokens(tokens);
    for (std::uint32_t index = keep; index < pages_.size(); ++index) {
        const std::uint32_t page = pages_[index];
        if (registry_->references(page) != 1 || registry_->active_references(page) != 0 ||
            registry_->source_pins(page) != 0 || registry_->protected_columns(page) != 0 ||
            registry_->host_resident(page)) {
            return false;
        }
    }
    return true;
}

void KvAddressSpace::truncate_inactive_prefix(std::uint32_t tokens) {
    if (released_) { throw std::logic_error("KV address space is released"); }
    if (!can_truncate_inactive_prefix(tokens)) {
        throw std::logic_error("KV address space suffix is still referenced or protected");
    }
    const std::uint32_t keep = pages_for_tokens(tokens);
    for (std::uint32_t index = keep; index < pages_.size(); ++index) {
        registry_->release_reference(pages_[index]);
    }
    pages_.resize(keep);
    committed_frontier_  = std::min(committed_frontier_, tokens);
    checkpoint_frontier_ = std::min(checkpoint_frontier_, tokens);
}

bool KvAddressSpace::can_release() const {
    for (const std::uint32_t page : pages_) {
        if (!registry_->can_release_reference(page)) { return false; }
    }
    return true;
}

void KvAddressSpace::release() {
    if (released_) { return; }
    released_ = true;
    for (const std::uint32_t page : pages_) { registry_->release_reference(page); }
    pages_.clear();
    committed_frontier_  = 0;
    checkpoint_frontier_ = 0;
}

} // namespace ninfer::targets::qwen3_6::detail
