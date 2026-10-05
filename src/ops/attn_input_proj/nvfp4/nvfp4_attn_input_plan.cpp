#include "ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.h"

#include "ops/linear/nvfp4/nvfp4_config.h"
#include "ops/linear/nvfp4/nvfp4_w4a4_plan.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

enum class Nvfp4AttnInputRoute : std::uint8_t {
    A16,
    W4A4,
};

Nvfp4AttnInputRoute resolve_route(LinearPolicy policy, std::int32_t tokens) {
    if (tokens <= 0) { throw std::invalid_argument("nvfp4 attn_input_proj: T must be positive"); }
    if (policy == LinearPolicy::A16Only) { return Nvfp4AttnInputRoute::A16; }
    if (policy != LinearPolicy::AllowA4) {
        throw std::invalid_argument("nvfp4 attn_input_proj: unsupported policy");
    }
    return tokens >= 4 ? Nvfp4AttnInputRoute::W4A4 : Nvfp4AttnInputRoute::A16;
}

void launch_a16(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                Tensor& v, WorkspaceArena* workspace, cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    nvfp4_attn_input_sm70_launch(x, weight, q, gate, k, v, workspace, stream);
#else
    (void)workspace;
    constexpr std::int32_t kChunk  = kNvfp4LastSmallT;
    constexpr std::int32_t kQRows  = 6144;
    constexpr std::int32_t kKvRows = 1024;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* query = static_cast<std::uint8_t*>(q.data) +
                      static_cast<std::int64_t>(token_begin) * kQRows * sizeof(std::uint16_t);
        auto* output_gate = static_cast<std::uint8_t*>(gate.data) +
                            static_cast<std::int64_t>(token_begin) * kQRows * sizeof(std::uint16_t);
        auto* key = static_cast<std::uint8_t*>(k.data) +
                    static_cast<std::int64_t>(token_begin) * kKvRows * sizeof(std::uint16_t);
        auto* value = static_cast<std::uint8_t*>(v.data) +
                      static_cast<std::int64_t>(token_begin) * kKvRows * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor query_chunk(query, DType::BF16, {kQRows, active});
        Tensor gate_chunk(output_gate, DType::BF16, {kQRows, active});
        Tensor key_chunk(key, DType::BF16, {kKvRows, active});
        Tensor value_chunk(value, DType::BF16, {kKvRows, active});
        if (active == 1) {
            nvfp4_attn_input_decode_launch(input_chunk, weight, query_chunk, gate_chunk, key_chunk,
                                           value_chunk, stream);
        } else {
            nvfp4_attn_input_small_t_launch(input_chunk, weight, query_chunk, gate_chunk, key_chunk,
                                            value_chunk, stream);
        }
    }
#endif
}

void launch_a16_shard(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate, Tensor& k,
                      Tensor& v, WorkspaceArena* workspace, cudaStream_t stream) {
#ifdef NINFER_VOLTA_BUILD
    nvfp4_attn_input_sm70_launch(x, weight, q, gate, k, v, workspace, stream);
#else
    (void)workspace;
    constexpr std::int32_t kChunk  = kNvfp4LastSmallT;
    constexpr std::int32_t kQRows  = Nvfp4AttnInputSections<Nvfp4AttnInputTp2ColumnGeometry>::kQueryRows;
    constexpr std::int32_t kKvRows = Nvfp4AttnInputSections<Nvfp4AttnInputTp2ColumnGeometry>::kKeyRows;
    for (std::int32_t token_begin = 0; token_begin < x.ne[1]; token_begin += kChunk) {
        const std::int32_t active = std::min(kChunk, x.ne[1] - token_begin);
        auto* input               = static_cast<std::uint8_t*>(x.data) +
                      static_cast<std::int64_t>(token_begin) * weight.k * sizeof(std::uint16_t);
        auto* query = static_cast<std::uint8_t*>(q.data) +
                      static_cast<std::int64_t>(token_begin) * kQRows * sizeof(std::uint16_t);
        auto* output_gate = static_cast<std::uint8_t*>(gate.data) +
                            static_cast<std::int64_t>(token_begin) * kQRows * sizeof(std::uint16_t);
        auto* key = static_cast<std::uint8_t*>(k.data) +
                    static_cast<std::int64_t>(token_begin) * kKvRows * sizeof(std::uint16_t);
        auto* value = static_cast<std::uint8_t*>(v.data) +
                      static_cast<std::int64_t>(token_begin) * kKvRows * sizeof(std::uint16_t);
        Tensor input_chunk(input, DType::BF16, {weight.k, active});
        Tensor query_chunk(query, DType::BF16, {kQRows, active});
        Tensor gate_chunk(output_gate, DType::BF16, {kQRows, active});
        Tensor key_chunk(key, DType::BF16, {kKvRows, active});
        Tensor value_chunk(value, DType::BF16, {kKvRows, active});
        if (active == 1) {
            nvfp4_attn_input_decode_launch_shard(input_chunk, weight, query_chunk, gate_chunk,
                                                 key_chunk, value_chunk, stream);
        } else {
            nvfp4_attn_input_small_t_launch_shard(input_chunk, weight, query_chunk, gate_chunk,
                                                  key_chunk, value_chunk, stream);
        }
    }
#endif
}

} // namespace

std::size_t nvfp4_attn_input_workspace_capacity_bytes(std::int32_t output_rows,
                                                      LinearPolicy policy, std::int32_t min_tokens,
                                                      std::int32_t max_tokens) {
    if ((output_rows != 14336 && output_rows != 7168) ||
        min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("nvfp4 attn_input_proj workspace: invalid token interval");
    }
    (void)resolve_route(policy, min_tokens);
    if (resolve_route(policy, max_tokens) == Nvfp4AttnInputRoute::W4A4) {
        return nvfp4_w4a4_workspace_capacity_bytes(max_tokens, Nvfp4AttnInputGeometry::kInputRows);
    }
#ifdef NINFER_VOLTA_BUILD
    return nvfp4_attn_input_sm70_workspace_bytes(output_rows, max_tokens);
#else
    return 0;
#endif
}

void nvfp4_attn_input_dispatch(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                               Tensor& k, Tensor& v, LinearPolicy policy, WorkspaceArena* workspace,
                               cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Nvfp4AttnInputRoute::A16) {
        launch_a16(x, weight, q, gate, k, v, workspace, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 attn_input_proj requires caller workspace");
    }
    auto scope                       = workspace->scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_attn_input_w4a4_launch(x, weight, q, gate, k, v, scratch, stream);
}

// --- TP2 column-shard sibling --------------------------------------------------------------------
// Route selection (resolve_route) is a pure function of (policy, token count), inherited unchanged
// from the tp1 parent; only the underlying kernel Geometry (and its section offsets) differ. The
// A4 scratch depends only on (T,K); wide SM70 A16 scratch also includes the shard output width.
void nvfp4_attn_input_dispatch_shard(const Tensor& x, const Weight& weight, Tensor& q, Tensor& gate,
                                     Tensor& k, Tensor& v, LinearPolicy policy,
                                     WorkspaceArena* workspace, cudaStream_t stream) {
    if (resolve_route(policy, x.ne[1]) == Nvfp4AttnInputRoute::A16) {
        launch_a16_shard(x, weight, q, gate, k, v, workspace, stream);
        return;
    }
    if (workspace == nullptr) {
        throw std::invalid_argument("nvfp4 W4A4 attn_input_proj column-parallel requires caller "
                                    "workspace");
    }
    auto scope                       = workspace->scope();
    const Nvfp4W4a4Workspace scratch = allocate_nvfp4_w4a4_workspace(*workspace, x.ne[1], weight.k);
    nvfp4_attn_input_w4a4_launch_shard(x, weight, q, gate, k, v, scratch, stream);
}

} // namespace ninfer::ops::detail
