// Bit-exact transfer of one LinearAttentionStatePool slot to/from a pinned host block.
//
// The pool stores each layer's conv/recurrent tensors with the slot as the FASTEST dim
// (last axis), so one slot's data is a strided slice: element i of slot s lives at
// `base[i * slot_stride + s]`. The host block is the slot's data repacked layer-major and
// contiguous:
//
//   [conv:      layers x (conv_channels x conv_width)                        conv_dtype]
//   [recurrent: layers x (key_head_dim x value_head_dim x value_heads)       FP32]
//
// The copy is an identity transform (no dtype change, no arithmetic), qualified against a
// naive per-element reference copy.

#include "core/linear_attention_state.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace ninfer {
namespace {

constexpr int kThreads = 256;
constexpr int kMaxGrid = 1024;

// The slot is a strided slice of the layer tensor (element i lives at i * slot_stride + slot)
// and contiguous in the host block, so the two directions are transposes of each other:
//   D2H: dst[i]                 = src[i * slot_stride + slot]  (strided device, contiguous host)
//   H2D: dst[i * slot_stride + s] = src[i]                     (contiguous host, strided device)
template <class Elem>
__global__ void linear_slot_transfer_kernel(const Elem* __restrict__ src, std::int64_t slot_stride,
                                            std::int64_t slot, std::int64_t total,
                                            Elem* __restrict__ dst, bool host_to_device) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < total; i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        if (host_to_device) {
            dst[i * slot_stride + slot] = src[i];
        } else {
            dst[i] = src[i * slot_stride + slot];
        }
    }
}

template <class Elem>
void launch_slot_transfer(const Elem* src, std::int64_t slot_stride, std::int64_t slot,
                          std::int64_t total, Elem* dst, bool host_to_device, cudaStream_t stream) {
    const int grid = static_cast<int>(
        std::min<std::int64_t>(kMaxGrid, (total + kThreads - 1) / kThreads));
    linear_slot_transfer_kernel<Elem><<<grid, kThreads, 0, stream>>>(
        src, slot_stride, slot, total, dst, host_to_device);
    CUDA_CHECK(cudaGetLastError());
}

std::int64_t checked_extent_product(std::int64_t a, std::int64_t b, const char* label) {
    if (a < 0 || b < 0 || (a != 0 && b > std::numeric_limits<std::int64_t>::max() / a)) {
        throw std::overflow_error(std::string(label) + " extent overflow");
    }
    return a * b;
}

} // namespace

[[nodiscard]] std::size_t linear_state_slot_bytes(const LinearAttentionStatePoolSpec& spec) {
    const std::int64_t conv_elements = checked_extent_product(
        checked_extent_product(spec.layers, spec.conv_channels, "conv"), spec.conv_width,
        "conv");
    const std::int64_t recurrent_elements = checked_extent_product(
        checked_extent_product(
            checked_extent_product(spec.layers, spec.key_head_dim, "recurrent"),
            spec.value_head_dim, "recurrent"),
        spec.value_heads, "recurrent");
    const std::int64_t bytes =
        conv_elements * dtype_size(spec.conv_dtype) + recurrent_elements * dtype_size(DType::FP32);
    if (bytes < 0 || static_cast<std::uint64_t>(bytes) >
                        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("linear state slot byte size overflow");
    }
    return static_cast<std::size_t>(bytes);
}

[[nodiscard]] std::size_t linear_state_conv_bytes(const LinearAttentionStatePoolSpec& spec) {
    const std::int64_t bytes = checked_extent_product(
        checked_extent_product(
            checked_extent_product(spec.layers, spec.conv_channels, "conv"), spec.conv_width,
            "conv"),
        dtype_size(spec.conv_dtype), "conv");
    if (bytes < 0 || static_cast<std::uint64_t>(bytes) >
                        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error("linear state conv byte size overflow");
    }
    return static_cast<std::size_t>(bytes);
}

namespace {

void transfer_slot(const LinearAttentionStatePool& pool, std::int32_t slot, void* host,
                   bool device_to_host, cudaStream_t stream) {
    const LinearAttentionStateAllLayersView view = pool.all_layers_view();
    const LinearAttentionStatePoolSpec& spec = view.spec;
    const std::int64_t conv_elements =
        static_cast<std::int64_t>(spec.conv_channels) * spec.conv_width;
    const std::int64_t recurrent_elements = static_cast<std::int64_t>(spec.key_head_dim) *
                                            spec.value_head_dim * spec.value_heads;
    const std::int64_t layers = spec.layers;
    const std::int64_t slot_index = slot;
    // The slot is the fastest axis of each layer tensor, so element i of slot s lives at
    // i * slot_count + s.
    const std::int64_t slot_stride = pool.slot_count();

    auto run_conv = [&](auto tag) {
        using Elem = decltype(tag);
        // Conv component: one launch per layer (the layer tensors are separate allocations).
        Elem* device_conv = reinterpret_cast<Elem*>(view.conv_layer0.data);
        Elem* host_conv   = reinterpret_cast<Elem*>(host);
        for (std::int64_t layer = 0; layer < layers; ++layer) {
            Elem* device_layer =
                reinterpret_cast<Elem*>(reinterpret_cast<std::byte*>(device_conv) +
                                        layer * view.conv_layer_stride_bytes);
            Elem* host_layer = host_conv + layer * conv_elements;
            if (device_to_host) {
                launch_slot_transfer<Elem>(device_layer, slot_stride, slot_index,
                                           conv_elements, host_layer, /*host_to_device=*/false,
                                           stream);
            } else {
                launch_slot_transfer<Elem>(host_layer, slot_stride, slot_index,
                                           conv_elements, device_layer, /*host_to_device=*/true,
                                           stream);
            }
        }
    };

    auto run_rec = [&]() {
        // Recurrent state is always FP32, after the conv block in the host buffer.
        float* device_rec = reinterpret_cast<float*>(view.recurrent_layer0.data);
        float* host_rec   = reinterpret_cast<float*>(reinterpret_cast<std::byte*>(host) +
                                                     linear_state_conv_bytes(spec));
        for (std::int64_t layer = 0; layer < layers; ++layer) {
            float* device_layer = reinterpret_cast<float*>(
                reinterpret_cast<std::byte*>(device_rec) + layer * view.recurrent_layer_stride_bytes);
            float* host_layer = host_rec + layer * recurrent_elements;
            if (device_to_host) {
                launch_slot_transfer<float>(device_layer, slot_stride, slot_index,
                                            recurrent_elements, host_layer,
                                            /*host_to_device=*/false, stream);
            } else {
                launch_slot_transfer<float>(host_layer, slot_stride, slot_index,
                                            recurrent_elements, device_layer,
                                            /*host_to_device=*/true, stream);
            }
        }
    };

    if (spec.conv_dtype == DType::BF16) {
        run_conv(std::uint16_t{});
    } else {
        run_conv(0.0F);
    }
    run_rec();
}

} // namespace

void snapshot_linear_state_slot_to_host(const LinearAttentionStatePool& pool, std::int32_t slot,
                                        void* host, cudaStream_t stream) {
    if (slot < 0 || slot >= pool.slot_count()) {
        throw std::out_of_range("linear state snapshot slot out of range");
    }
    transfer_slot(pool, slot, host, /*device_to_host=*/true, stream);
}

void restore_linear_state_slot_from_host(const LinearAttentionStatePool& pool, std::int32_t slot,
                                         const void* host, cudaStream_t stream) {
    if (slot < 0 || slot >= pool.slot_count()) {
        throw std::out_of_range("linear state restore slot out of range");
    }
    transfer_slot(pool, slot, const_cast<void*>(host), /*device_to_host=*/false, stream);
}

} // namespace ninfer