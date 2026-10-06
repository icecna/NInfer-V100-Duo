#pragma once

// Compact host identity for the model inputs licensed by the resident KV/GDN state.

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace ninfer::targets::qwen3_6::detail {

// Rolling per-frontier content digest. It summarizes the same identity fields the exact comparison
// uses (token id, token type, three-axis position, Vision items) so it can shortlist checkpoints
// before the authoritative prefix_matches runs.
using PrefixDigest = std::array<std::uint64_t, 2>;

// Semantic shortlist key for one prefix frontier. Two keys are comparable only when their
// identity_tag matches (same speculative backend, KV storage, TP width, and vision mode); the
// digest is a filter and never authorizes reuse by itself.
struct PrefixShortlistKey {
    PrefixDigest digest{};
    std::uint32_t frontier     = 0;
    std::uint32_t identity_tag = 0;

    [[nodiscard]] friend constexpr bool operator==(const PrefixShortlistKey&,
                                                   const PrefixShortlistKey&) noexcept = default;
};

// Pack the program-fixed identity of a request plan. A checkpoint captured under one speculative
// backend / KV storage / TP width / vision mode must never be reused by another program.
[[nodiscard]] std::uint32_t make_prefix_identity_tag(std::uint32_t speculative_backend,
                                                     std::uint32_t proposal_head,
                                                     std::uint32_t kv_dtype,
                                                     std::int32_t kv_quant_group, int tp,
                                                     bool vision) noexcept;

// Compute the rolling digest of every prompt frontier. `out` is sized
// prompt.token_ids.size() + 1; out[frontier] covers tokens [0, frontier) and out[0] is the seed.
void assign_prefix_digests(const PreparedPromptData& prompt, std::vector<PrefixDigest>& out);

class ResidentPrefixIdentity {
public:
    void reserve(std::size_t tokens);
    void clear() noexcept;
    void assign(const PreparedPromptData& prompt);
    void append_generated(std::size_t count, std::int32_t rope_delta);
    void truncate(std::size_t tokens);

    [[nodiscard]] std::size_t size() const noexcept { return token_types_.size(); }

    [[nodiscard]] bool matches(const PreparedPromptData& prompt, std::size_t count) const;

    // Digest of the first `frontier` tokens. Requires frontier <= size().
    [[nodiscard]] const PrefixDigest& digest_at(std::size_t frontier) const;

private:
    std::vector<std::uint8_t> token_types_;
    std::array<std::vector<std::int32_t>, 3> positions_;
    std::vector<VisionItem> vision_items_;
    std::vector<PrefixDigest> digests_;
};

[[nodiscard]] bool prefix_matches(const PreparedPromptData& prompt,
                                  const std::vector<TokenId>& resident_tokens,
                                  const ResidentPrefixIdentity& resident_identity,
                                  std::size_t count);

} // namespace ninfer::targets::qwen3_6::detail
