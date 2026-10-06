// Qualification of the bit-exact LinearAttentionStatePool slot transfer
// (snapshot_linear_state_slot_to_host / restore_linear_state_slot_from_host).
//
// The transfer is an identity transform with no arithmetic, so the oracle is a naive
// per-element reference copy of the strided slot slice: element i of slot s lives at
// `base[i * slot_stride + s]` on device and `base[i]` in the host block. Both directions are
// byte-exact checks; any byte drift fails.

#include "core/linear_attention_state.h"
#include "core/layout.h"
#include "core/tensor.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace ninfer;

int failures = 0;

int fail(const std::string& label) {
    std::cerr << label << '\n';
    ++failures;
    return failures;
}

int expect(bool condition, const std::string& label) { return condition ? 0 : fail(label); }

int expect_throws(std::function<void()> fn, const std::string& label) {
    try {
        fn();
    } catch (const std::exception&) { return 0; }
    return fail(label + " did not throw");
}

// Deterministic byte pattern: one LCG byte stream per test run, indexed by the global byte
// offset inside the pool backing.
std::uint32_t lcg_state = 0x9E3779B9u;
std::byte pattern_byte(std::size_t offset) {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return static_cast<std::byte>(lcg_state >> 16);
}

std::size_t align_up(std::size_t value, std::size_t alignment) {
    return (value + alignment - 1) / alignment * alignment;
}

// Fill the device backing with the byte pattern (dtype-blind: the reference is over bytes).
void fill_pattern(void* device, std::size_t bytes) {
    std::vector<std::byte> host(bytes);
    for (std::size_t i = 0; i < bytes; ++i) { host[i] = pattern_byte(i); }
    if (cudaMemcpy(device, host.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        throw std::runtime_error("pattern fill copy failed");
    }
}

int run_directions(const LinearAttentionStatePoolSpec& spec, std::uint32_t seed) {
    lcg_state = seed;
    LayoutBuilder builder;
    const auto layout = plan_linear_attention_state_pool(builder, spec);
    const std::size_t bytes = builder.finish(256);

    void* device = nullptr;
    if (cudaMalloc(&device, bytes) != cudaSuccess) { return fail("cudaMalloc failed"); }
    std::byte* device_bytes = static_cast<std::byte*>(device);

    fill_pattern(device, bytes);
    // The reference image of the whole backing, taken BEFORE any transfer.
    std::vector<std::byte> reference(bytes);
    if (cudaMemcpy(reference.data(), device, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaFree(device);
        return fail("reference copy failed");
    }

    LinearAttentionStatePool pool({device, bytes}, layout);
    const auto view = pool.all_layers_view();
    const std::int64_t conv_elem = static_cast<std::int64_t>(spec.conv_channels) * spec.conv_width;
    const std::int64_t rec_elem = static_cast<std::int64_t>(spec.key_head_dim) *
                                  spec.value_head_dim * spec.value_heads;
    const std::int64_t layers = spec.layers;
    const std::int64_t slots = spec.slot_count;
    const std::size_t conv_bytes = linear_state_conv_bytes(spec);
    const std::size_t slot_bytes = linear_state_slot_bytes(spec);
    const std::size_t rec_bytes = conv_bytes ? slot_bytes - conv_bytes : slot_bytes;

    if (expect_throws(
            [&] { snapshot_linear_state_slot_to_host(pool, slots, device, nullptr); },
            "snapshot slot above the pool") != 0 ||
        expect_throws([&] { snapshot_linear_state_slot_to_host(pool, -1, device, nullptr); },
                      "snapshot negative slot") != 0) {
        cudaFree(device);
        return failures;
    }

    // Pinned, 256-aligned host block.
    std::byte* raw_host = nullptr;
    if (cudaMallocHost(reinterpret_cast<void**>(&raw_host), slot_bytes + 256) != cudaSuccess) {
        cudaFree(device);
        return fail("cudaMallocHost failed");
    }
    std::byte* host_block = reinterpret_cast<std::byte*>(
        (reinterpret_cast<std::uintptr_t>(raw_host) + 255) & ~std::uintptr_t{255});

    int local_failures = failures;
    for (const std::int32_t slot : {0, spec.slot_count - 1, (std::int32_t)(seed & 1) + 1}) {
        // ---- D2H: snapshot, then byte-exact compare against the naive reference. ----
        std::vector<std::byte> expected(slot_bytes, std::byte{0});
        // The reference image (host-side copy of the whole backing) is the source of truth:
        // device tensor byte offsets are constant offsets into the backing.
        const std::size_t conv_layer_stride = static_cast<std::size_t>(
            static_cast<std::int64_t>(view.conv_layer_stride_bytes));
        const std::size_t rec_layer_stride = static_cast<std::size_t>(static_cast<std::int64_t>(
            view.recurrent_layer_stride_bytes));
        const std::size_t conv_base = static_cast<std::size_t>(
            static_cast<const std::byte*>(view.conv_layer0.data) - device_bytes);
        const std::size_t rec_base = static_cast<std::size_t>(
            static_cast<const std::byte*>(view.recurrent_layer0.data) - device_bytes);
        {
            for (std::int64_t l = 0; l < layers; ++l) {
                const std::byte* layer = reference.data() + conv_base + l * conv_layer_stride;
                for (std::int64_t i = 0; i < conv_elem; ++i) {
                    const std::byte* src =
                        layer + static_cast<std::size_t>(i * slots + slot) *
                                    dtype_size(spec.conv_dtype);
                    std::byte* dst =
                        expected.data() + static_cast<std::size_t>(l * conv_elem + i) *
                                              dtype_size(spec.conv_dtype);
                    std::memcpy(dst, src, dtype_size(spec.conv_dtype));
                }
            }
            for (std::int64_t l = 0; l < layers; ++l) {
                const std::byte* layer = reference.data() + rec_base + l * rec_layer_stride;
                for (std::int64_t i = 0; i < rec_elem; ++i) {
                    const std::byte* src =
                        layer + static_cast<std::size_t>(i * slots + slot) * dtype_size(DType::FP32);
                    std::byte* dst = expected.data() + conv_bytes +
                                     static_cast<std::size_t>(l * rec_elem + i) *
                                         dtype_size(DType::FP32);
                    std::memcpy(dst, src, dtype_size(DType::FP32));
                }
            }
        }

        snapshot_linear_state_slot_to_host(pool, slot, host_block, nullptr);
        if (cudaStreamSynchronize(nullptr) != cudaSuccess) { std::fprintf(stderr, "D2H sync: %s\n", cudaGetErrorString(cudaGetLastError())); cudaFreeHost(raw_host); cudaFree(device); return fail("D2H sync failed"); }
        if (std::memcmp(host_block, expected.data(), slot_bytes) != 0) {
            std::fprintf(stderr, "D2H slot=%d conv_dtype=%s mismatch (first byte drift below)\n",
                         slot, spec.conv_dtype == DType::BF16 ? "BF16" : "FP32");
            int shown = 0;
            for (std::size_t i = 0; i < slot_bytes && shown < 8; ++i) {
                if (host_block[i] != expected[i]) {
                    std::fprintf(stderr, "  offset %zu: got %02x expected %02x\n", i,
                                 static_cast<unsigned char>(host_block[i]),
                                 static_cast<unsigned char>(expected[i]));
                    ++shown;
                }
            }
            cudaFreeHost(raw_host);
            cudaFree(device);
            return fail("D2H snapshot not byte-exact");
        }

        // ---- H2D: clobber the slot elements on device, restore, whole-backing byte compare.
        std::vector<std::byte> clobbered = reference;
        {
            const std::byte* dev_conv = static_cast<const std::byte*>(view.conv_layer0.data);
            const std::size_t conv_layer_stride = static_cast<std::size_t>(
                static_cast<std::int64_t>(view.conv_layer_stride_bytes));
            for (std::int64_t l = 0; l < layers; ++l) {
                std::byte* layer =
                    clobbered.data() + (dev_conv - device_bytes) + l * conv_layer_stride;
                for (std::int64_t i = 0; i < conv_elem; ++i) {
                    std::byte* p = layer + static_cast<std::size_t>(i * slots + slot) *
                                             dtype_size(spec.conv_dtype);
                    std::memset(p, 0xAB, dtype_size(spec.conv_dtype));
                }
            }
            const std::byte* dev_rec = static_cast<const std::byte*>(view.recurrent_layer0.data);
            const std::size_t rec_layer_stride = static_cast<std::size_t>(static_cast<std::int64_t>(
                view.recurrent_layer_stride_bytes));
            for (std::int64_t l = 0; l < layers; ++l) {
                std::byte* layer =
                    clobbered.data() + (dev_rec - device_bytes) + l * rec_layer_stride;
                for (std::int64_t i = 0; i < rec_elem; ++i) {
                    std::byte* p = layer + static_cast<std::size_t>(i * slots + slot) *
                                             dtype_size(DType::FP32);
                    std::memset(p, 0xCD, dtype_size(DType::FP32));
                }
            }
            // Sanity: the clobber actually touched memory (slot elements differ from the ref).
            bool touched = false;
            for (std::size_t i = 0; i < clobbered.size(); ++i) {
                if (clobbered[i] != reference[i]) { touched = true; break; }
            }
            if (!touched) { cudaFreeHost(raw_host); cudaFree(device); return fail("clobber had no effect"); }
        }
        if (cudaMemcpy(device, clobbered.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFreeHost(raw_host);
            cudaFree(device);
            return fail("clobber copy failed");
        }
        restore_linear_state_slot_from_host(pool, slot, host_block, nullptr);
        if (cudaStreamSynchronize(nullptr) != cudaSuccess) { std::fprintf(stderr, "H2D sync: %s\n", cudaGetErrorString(cudaGetLastError())); cudaFreeHost(raw_host); cudaFree(device); return fail("H2D sync failed"); }
        std::vector<std::byte> restored(bytes);
        if (cudaMemcpy(restored.data(), device, bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaFreeHost(raw_host);
            cudaFree(device);
            return fail("restored copy failed");
        }
        if (std::memcmp(restored.data(), reference.data(), bytes) != 0) {
            std::fprintf(stderr, "H2D slot=%d conv_dtype=%s restored backing differs\n", slot,
                         spec.conv_dtype == DType::BF16 ? "BF16" : "FP32");
            cudaFreeHost(raw_host);
            cudaFree(device);
            return fail("H2D restore not byte-exact");
        }
    }
    cudaFreeHost(raw_host);
    cudaFree(device);
    if (failures != local_failures) {
        std::cerr << "linear state transfer (conv_dtype "
                  << (spec.conv_dtype == DType::BF16 ? "BF16" : "FP32") << ") failed\n";
    }
    return failures;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cerr << "no CUDA device; skipping\n";
        return 77;
    }
    if (cudaSetDevice(0) != cudaSuccess) { return 1; }

    // Small pool covering both conv dtypes, multiple layers and slots; the 27b production
    // shape (48 layers, 5120 conv channels) scales the same code path.
    const LinearAttentionStatePoolSpec bf16_spec{
        .layers = 4, .conv_channels = 256, .conv_width = 3, .value_heads = 6,
        .value_head_dim = 128, .key_head_dim = 128, .slot_count = 9, .conv_dtype = DType::BF16};
    const LinearAttentionStatePoolSpec fp32_spec{
        .layers = 3, .conv_channels = 64, .conv_width = 4, .value_heads = 2,
        .value_head_dim = 32, .key_head_dim = 32, .slot_count = 5, .conv_dtype = DType::FP32};

    (void)run_directions(bf16_spec, 12345u);
    (void)run_directions(fp32_spec, 67890u);

    if (failures == 0) { std::cout << "ok\n"; return 0; }
    std::cerr << failures << " linear state transfer check(s) failed\n";
    return 1;
}