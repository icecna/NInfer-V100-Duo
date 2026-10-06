#pragma once

// Cost model that ranks context-cache candidates and decides whether reclaiming one resource for
// another is worth it. Prices are fixed-point Q32 nanoseconds so they stay exact across hosts and
// comparable without floating point. A transfer price is the larger of a launch-dominated estimate
// (batch + per-copy overhead) and a byte-dominated estimate (payload bytes / bandwidth); a prefill
// price is a chunk term plus per-token, per-attention-pair, and per-vision-item/patch terms.

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace ninfer::runtime {

inline constexpr std::uint64_t kContextCostQ32One = 1ULL << 32;

[[nodiscard]] constexpr std::uint64_t scale_q32(std::uint64_t value_q32,
                                                std::uint64_t count) noexcept {
    // Split the multiply so a large count cannot overflow 64 bits.
    const std::uint64_t whole = value_q32 >> 32;
    const std::uint64_t frac  = value_q32 & (kContextCostQ32One - 1U);
    return whole * count + ((frac * count) >> 32);
}

struct ContextTransferCost {
    // Fixed launch cost of one batched transfer, plus one per copy operation, plus a per-byte term.
    std::uint64_t batch_ns       = 0;
    std::uint64_t operation_ns   = 0;
    std::uint64_t ns_per_byte_q32 = 0;

    [[nodiscard]] constexpr std::uint64_t price_ns(std::size_t payload_bytes,
                                                   std::uint32_t operations) const noexcept {
        const std::uint64_t launch =
            batch_ns + static_cast<std::uint64_t>(operation_ns) * operations;
        const std::uint64_t bytes = scale_q32(ns_per_byte_q32, payload_bytes);
        return std::max(launch, bytes);
    }
};

struct ContextPrefillCost {
    std::uint64_t chunk_ns              = 0;
    std::uint64_t token_ns_q32          = 0;
    std::uint64_t attention_pair_ns_q32 = 0;
    std::uint64_t vision_item_ns        = 0;
    std::uint64_t vision_patch_ns_q32   = 0;

    [[nodiscard]] constexpr std::uint64_t price_ns(std::uint32_t tokens, std::uint64_t attention_pairs,
                                                   std::uint32_t vision_items,
                                                   std::uint64_t vision_patches) const noexcept {
        return chunk_ns + scale_q32(token_ns_q32, tokens) +
               scale_q32(attention_pair_ns_q32, attention_pairs) +
               static_cast<std::uint64_t>(vision_item_ns) * vision_items +
               scale_q32(vision_patch_ns_q32, vision_patches);
    }
};

struct ContextMachineCostModel {
    ContextTransferCost device_to_host;
    ContextTransferCost host_to_device;
    ContextTransferCost device_to_device;
    ContextPrefillCost prefill;
};

} // namespace ninfer::runtime
