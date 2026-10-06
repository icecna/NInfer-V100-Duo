#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6::detail {
namespace {

constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
constexpr PrefixDigest kDigestSeed{0x243F6A8885A308D3ULL, 0x13198A2E03707344ULL};
constexpr std::uint64_t kTokenDigestDomain     = 0x9E3779B97F4A7C15ULL;
constexpr std::uint64_t kGeneratedDigestDomain = 0xD1B54A32D192ED03ULL;
constexpr std::uint64_t kVisionDigestDomain    = 0x94D049BB133111EBULL;

void mix(PrefixDigest& digest, std::uint64_t value) {
    digest[0] ^= value;
    digest[0] *= kFnvPrime;
    digest[1] ^= value + 0x9E3779B97F4A7C15ULL;
    digest[1] *= kFnvPrime;
}

void mix_token(PrefixDigest& digest, std::int32_t id, std::uint8_t type, std::int32_t temporal,
               std::int32_t height, std::int32_t width) {
    mix(digest, kTokenDigestDomain);
    mix(digest, static_cast<std::uint32_t>(id));
    mix(digest, type);
    mix(digest, static_cast<std::uint32_t>(temporal));
    mix(digest, static_cast<std::uint32_t>(height));
    mix(digest, static_cast<std::uint32_t>(width));
}

void mix_vision_item(PrefixDigest& digest, const VisionItem& item) {
    mix(digest, kVisionDigestDomain);
    mix(digest, static_cast<std::uint32_t>(item.modality));
    mix(digest, static_cast<std::uint32_t>(item.grid.temporal));
    mix(digest, static_cast<std::uint32_t>(item.grid.height));
    mix(digest, static_cast<std::uint32_t>(item.grid.width));
    mix(digest, item.patch_begin);
    mix(digest, item.patch_count);
    for (const std::uint8_t byte : item.content_digest) { mix(digest, byte); }
    for (const double timestamp : item.timestamps) {
        mix(digest, std::bit_cast<std::uint64_t>(timestamp));
    }
    mix(digest, item.token_spans.size());
    for (const TokenSpan& span : item.token_spans) {
        mix(digest, span.begin);
        mix(digest, span.count);
    }
}

// Frontier (token count) after which a Vision item is fully consumed.
std::size_t vision_item_frontier(const VisionItem& item) {
    if (item.token_spans.empty()) { return 0; }
    const TokenSpan& last = item.token_spans.back();
    return last.begin + last.count;
}

bool same_grid(const VisionGrid& left, const VisionGrid& right) {
    return left.temporal == right.temporal && left.height == right.height &&
           left.width == right.width;
}

bool same_spans(const std::vector<TokenSpan>& left, const std::vector<TokenSpan>& right) {
    return left.size() == right.size() && std::equal(left.begin(), left.end(), right.begin(),
                                                     [](const TokenSpan& a, const TokenSpan& b) {
                                                         return a.begin == b.begin &&
                                                                a.count == b.count;
                                                     });
}

bool same_item(const VisionItem& left, const VisionItem& right) {
    return left.modality == right.modality && same_grid(left.grid, right.grid) &&
           left.patch_begin == right.patch_begin && left.patch_count == right.patch_count &&
           left.content_digest == right.content_digest && left.timestamps == right.timestamps &&
           same_spans(left.token_spans, right.token_spans);
}

bool prefix_item_count(const std::vector<VisionItem>& items, std::size_t tokens,
                       std::size_t* count) {
    *count          = 0;
    bool saw_suffix = false;
    for (const VisionItem& item : items) {
        if (item.token_spans.empty()) { return false; }
        const TokenSpan& first = item.token_spans.front();
        const TokenSpan& last  = item.token_spans.back();
        if (first.count == 0 || last.count == 0 ||
            last.begin > std::numeric_limits<std::size_t>::max() - last.count) {
            return false;
        }
        const std::size_t end = last.begin + last.count;
        if (end <= tokens) {
            if (saw_suffix) { return false; }
            ++*count;
        } else if (first.begin >= tokens) {
            saw_suffix = true;
        } else {
            // A reusable frontier may not divide the consumers of one Vision item.
            return false;
        }
    }
    return true;
}

} // namespace

std::uint32_t make_prefix_identity_tag(std::uint32_t speculative_backend,
                                       std::uint32_t proposal_head, std::uint32_t kv_dtype,
                                       std::int32_t kv_quant_group, int tp, bool vision) noexcept {
    std::uint32_t tag = 2166136261U;
    const auto mix32  = [&tag](std::uint32_t value) {
        tag ^= value;
        tag *= 16777619U;
    };
    mix32(speculative_backend);
    mix32(proposal_head);
    mix32(kv_dtype);
    mix32(static_cast<std::uint32_t>(kv_quant_group));
    mix32(static_cast<std::uint32_t>(tp));
    mix32(vision ? 1U : 0U);
    return tag;
}

void assign_prefix_digests(const PreparedPromptData& prompt, std::vector<PrefixDigest>& out) {
    const std::size_t tokens = prompt.token_ids.size();
    if (prompt.token_types.size() != tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("prepared prompt identity metadata has an invalid shape");
    }
    out.assign(tokens + 1, kDigestSeed);
    PrefixDigest running = kDigestSeed;

    // A Vision item is mixed once its consumer tokens are complete, so out[frontier] summarizes
    // exactly the media included in the prefix. An item with no consumer span is mixed at the seed.
    std::vector<std::size_t> item_frontiers(prompt.vision_items.size());
    for (std::size_t index = 0; index < item_frontiers.size(); ++index) {
        item_frontiers[index] = vision_item_frontier(prompt.vision_items[index]);
    }
    std::size_t next_item = 0;
    while (next_item < item_frontiers.size() && item_frontiers[next_item] == 0) {
        mix_vision_item(running, prompt.vision_items[next_item]);
        ++next_item;
    }
    for (std::size_t index = 0; index < tokens; ++index) {
        mix_token(running, prompt.token_ids[index], prompt.token_types[index],
                  prompt.positions[index], prompt.positions[tokens + index],
                  prompt.positions[2 * tokens + index]);
        while (next_item < item_frontiers.size() && item_frontiers[next_item] == index + 1) {
            mix_vision_item(running, prompt.vision_items[next_item]);
            ++next_item;
        }
        out[index + 1] = running;
    }
}

void ResidentPrefixIdentity::reserve(std::size_t tokens) {
    token_types_.reserve(tokens);
    for (auto& axis : positions_) { axis.reserve(tokens); }
    digests_.reserve(tokens + 1);
}

void ResidentPrefixIdentity::clear() noexcept {
    token_types_.clear();
    for (auto& axis : positions_) { axis.clear(); }
    vision_items_.clear();
    digests_.clear();
}

void ResidentPrefixIdentity::assign(const PreparedPromptData& prompt) {
    const std::size_t tokens = prompt.token_ids.size();
    if (prompt.token_types.size() != tokens || prompt.positions.size() != 3 * tokens) {
        throw std::invalid_argument("prepared prompt identity metadata has an invalid shape");
    }
    token_types_ = prompt.token_types;
    for (std::size_t axis = 0; axis < positions_.size(); ++axis) {
        const auto begin = prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * tokens);
        positions_[axis].assign(begin, begin + static_cast<std::ptrdiff_t>(tokens));
    }
    vision_items_ = prompt.vision_items;
    assign_prefix_digests(prompt, digests_);
}

void ResidentPrefixIdentity::append_generated(std::size_t count, std::int32_t rope_delta) {
    const std::size_t begin = size();
    if (count > std::numeric_limits<std::size_t>::max() - begin) {
        throw std::overflow_error("generated prefix identity length overflows size_t");
    }
    if (digests_.size() != begin + 1) { digests_.assign(begin + 1, kDigestSeed); }
    for (std::size_t offset = 0; offset < count; ++offset) {
        const std::size_t index = begin + offset;
        if (index > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
            throw std::overflow_error("generated prefix position exceeds int32");
        }
        const std::int64_t position = static_cast<std::int64_t>(index) + rope_delta;
        if (position < std::numeric_limits<std::int32_t>::min() ||
            position > std::numeric_limits<std::int32_t>::max()) {
            throw std::overflow_error("generated MRoPE position exceeds int32");
        }
        token_types_.push_back(0);
        for (auto& axis : positions_) { axis.push_back(static_cast<std::int32_t>(position)); }
        // Generated ids are not part of the resident identity (they live in the ledger), so the
        // extended digest is a conservative filter only; exact matching still uses prefix_matches.
        PrefixDigest next = digests_.back();
        mix(next, kGeneratedDigestDomain);
        mix(next, 0U);
        mix(next, 0U);
        mix(next, static_cast<std::uint32_t>(position));
        mix(next, static_cast<std::uint32_t>(position));
        mix(next, static_cast<std::uint32_t>(position));
        digests_.push_back(next);
    }
}

void ResidentPrefixIdentity::truncate(std::size_t tokens) {
    if (tokens > size()) {
        throw std::out_of_range("cannot extend resident prefix identity by truncation");
    }
    std::size_t retained_items = 0;
    if (!prefix_item_count(vision_items_, tokens, &retained_items)) {
        throw std::logic_error("resident prefix truncation divides a Vision item");
    }
    token_types_.resize(tokens);
    for (auto& axis : positions_) { axis.resize(tokens); }
    vision_items_.resize(retained_items);
    digests_.resize(tokens + 1);
}

const PrefixDigest& ResidentPrefixIdentity::digest_at(std::size_t frontier) const {
    if (frontier >= digests_.size()) {
        throw std::out_of_range("prefix digest frontier exceeds resident identity");
    }
    return digests_[frontier];
}

bool ResidentPrefixIdentity::matches(const PreparedPromptData& prompt, std::size_t count) const {
    const std::size_t prompt_tokens = prompt.token_ids.size();
    if (count > prompt_tokens || count > size() || prompt.token_types.size() != prompt_tokens ||
        prompt.positions.size() != 3 * prompt_tokens) {
        return false;
    }
    if (!std::equal(prompt.token_types.begin(),
                    prompt.token_types.begin() + static_cast<std::ptrdiff_t>(count),
                    token_types_.begin())) {
        return false;
    }
    for (std::size_t axis = 0; axis < positions_.size(); ++axis) {
        const auto begin =
            prompt.positions.begin() + static_cast<std::ptrdiff_t>(axis * prompt_tokens);
        if (!std::equal(begin, begin + static_cast<std::ptrdiff_t>(count),
                        positions_[axis].begin())) {
            return false;
        }
    }

    std::size_t incoming_items = 0;
    std::size_t resident_items = 0;
    if (!prefix_item_count(prompt.vision_items, count, &incoming_items) ||
        !prefix_item_count(vision_items_, count, &resident_items) ||
        incoming_items != resident_items) {
        return false;
    }
    for (std::size_t i = 0; i < incoming_items; ++i) {
        if (!same_item(prompt.vision_items[i], vision_items_[i])) { return false; }
    }
    return true;
}

bool prefix_matches(const PreparedPromptData& prompt, const std::vector<TokenId>& resident_tokens,
                    const ResidentPrefixIdentity& resident_identity, std::size_t count) {
    if (count > prompt.token_ids.size() || count > resident_tokens.size()) { return false; }
    return std::equal(prompt.token_ids.begin(),
                      prompt.token_ids.begin() + static_cast<std::ptrdiff_t>(count),
                      resident_tokens.begin()) &&
           resident_identity.matches(prompt, count);
}

} // namespace ninfer::targets::qwen3_6::detail
