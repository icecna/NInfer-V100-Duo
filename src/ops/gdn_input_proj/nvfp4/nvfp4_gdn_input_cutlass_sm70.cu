#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_cutlass_sm70.h"

#include "core/device.h"
#include "core/layout.h"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_output.cuh"
#include "ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.h"
#include "ops/linear/nvfp4/nvfp4_cutlass_sm70.h"

#include <cuda_bf16.h>

namespace ninfer::ops::detail {
namespace {

// Wide-token A16 route on SM70. Same numeric profile as the attention sibling: one qualified
// NVFP4 CUTLASS GEMM into a workspace plane (FP32 accumulation, single BF16 rounding), then a
// split that distributes the already-rounded values into the qkv and z planes. The stored NVFP4
// codes are unchanged.

template <class Geometry, class Allocator>
Tensor allocate_projected(Allocator& allocator, int tokens) {
    return allocator.alloc(DType::BF16, {Geometry::kOutputRows, tokens});
}

template <class Geometry>
__global__ void split_gdn_output(const __nv_bfloat16* __restrict__ projected,
                                 Nvfp4GdnInputShardOutput<Geometry> output, std::int64_t count) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) { return; }
    const int row   = static_cast<int>(i % Geometry::kOutputRows);
    const int token = static_cast<int>(i / Geometry::kOutputRows);
    *output.destination(row, token) = projected[i];
}

template <class Geometry>
std::size_t workspace_bytes(std::int32_t tokens) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_projected<Geometry>(layout, tokens);
    return layout.peak_bytes(1) +
           nvfp4_cutlass_sm70_workspace_bytes(Geometry::kOutputRows, Geometry::kInputRows, tokens);
}

template <class Geometry>
void launch(const Tensor& x, const Weight& weight, Tensor& qkv, Tensor& z,
            WorkspaceArena& workspace, cudaStream_t stream) {
    auto scope       = workspace.scope();
    Tensor projected = allocate_projected<Geometry>(workspace, x.ne[1]);
    nvfp4_cutlass_sm70_launch(x, weight, projected, workspace, stream);
    const std::int64_t count = static_cast<std::int64_t>(x.ne[1]) * Geometry::kOutputRows;
    Nvfp4GdnInputShardOutput<Geometry> output{static_cast<__nv_bfloat16*>(qkv.data),
                                              static_cast<__nv_bfloat16*>(z.data)};
    split_gdn_output<Geometry><<<static_cast<unsigned>((count + 255) / 256), 256, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected.data), output, count);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

std::size_t nvfp4_gdn_input_cutlass_workspace_bytes(std::int32_t tokens) {
    return workspace_bytes<Nvfp4GdnInputGeometry>(tokens);
}

std::size_t nvfp4_gdn_input_cutlass_shard_workspace_bytes(std::int32_t tokens) {
    return workspace_bytes<Nvfp4GdnInputTp2ColumnGeometry>(tokens);
}

void nvfp4_gdn_input_cutlass_sm70_launch(const Tensor& x, const Weight& weight, Tensor& qkv,
                                         Tensor& z, WorkspaceArena& workspace,
                                         cudaStream_t stream) {
    launch<Nvfp4GdnInputGeometry>(x, weight, qkv, z, workspace, stream);
}

void nvfp4_gdn_input_cutlass_sm70_launch_shard(const Tensor& x, const Weight& weight, Tensor& qkv,
                                               Tensor& z, WorkspaceArena& workspace,
                                               cudaStream_t stream) {
    launch<Nvfp4GdnInputTp2ColumnGeometry>(x, weight, qkv, z, workspace, stream);
}

} // namespace ninfer::ops::detail