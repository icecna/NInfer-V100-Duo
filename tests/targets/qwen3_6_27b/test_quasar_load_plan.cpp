// Host-only load-plan qualification for the qwen3.8-27b/quasar-nvfp4 trial artifact.
//
// Independent source-file oracle: re-reads the published v3 file's own directory and checks
// that the projected reader inventory, every NVFP4 parent's exact original packed range, all
// 256 FP32 scale pairs, the W8 endpoints and MTP format, the BF16 GDN A/B boundary, and the
// text-only/Vision startup placements all agree with the source. No GPU allocation or weight
// conversion.
// Set NINFER_QUASAR_ARTIFACT to the quasar v3 file; the suite reports the skip code otherwise.
#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
using namespace ninfer::artifact;
using namespace ninfer::targets::qwen3_6_27b::detail;
using Json = nlohmann::json;

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

std::uint64_t little_endian(const char* bytes, int count) {
    std::uint64_t result = 0;
    for (int i = 0; i < count; ++i) {
        result |= std::uint64_t(static_cast<unsigned char>(bytes[i])) << (8 * i);
    }
    return result;
}

void verify(const char* path) {
    std::ifstream source(path, std::ios::binary);
    std::array<char, 32> header{};
    source.read(header.data(), header.size());
    require(source.good() && header[7] == 3, "QUASAR test requires its registered v3 file");
    std::string json_bytes(little_endian(header.data() + 8, 8), '\0');
    source.read(json_bytes.data(), json_bytes.size());
    require(source.good(), "truncated source directory");
    const Json directory = Json::parse(json_bytes);
    const auto payload_start = (header.size() + json_bytes.size() + 4095) / 4096 * 4096;
    std::unordered_map<std::string, Json> stored;
    for (const auto& object : directory.at("objects")) {
        stored.emplace(object.at("id").get<std::string>(), object);
    }
    const auto scalar = [&](std::uint64_t offset) {
        std::array<char, 4> bytes{};
        source.seekg(payload_start + offset);
        source.read(bytes.data(), bytes.size());
        require(source.good(), "source FP32 scalar is truncated");
        return static_cast<std::uint32_t>(little_endian(bytes.data(), 4));
    };

    Reader reader(path);
    require(reader.identity().model_id == "qwen3.8-27b" &&
                reader.identity().weights_id == "quasar-nvfp4", "QUASAR identity was not retained");
    // 1012 structural objects plus the 256 NVFP4 activation-divisor auxiliaries. The file's
    // DFlash2 companion objects are payload-only here: the trial admits no DFlash residency.
    require(reader.objects().size() == 1268, "QUASAR projected inventory is incomplete");
    for (const auto backend : {ninfer::SpeculativeBackend::None, ninfer::SpeculativeBackend::Mtp}) {
        Binder binder(reader, 2);
        const auto plan = bind_artifact(binder, WeightsProfile::Qwen38QuasarNvfp4,
                                        {.speculative = backend}, 2);
        int projections = 0;
        const auto check = [&](const WeightPlan& weight, const std::string& divisor) {
            const auto& descriptor = std::get<TensorDescriptor>(binder.descriptor(weight.object));
            const auto& original   = stored.at(descriptor.name);
            require(weight.format == NumericFormat::NVFP4 && original.at("format") == "nvfp4",
                    descriptor.name + ": wrong weight format");
            const auto payload = reader.payload(descriptor);
            require(payload.absolute_offset ==
                        payload_start + original.at("offset").get<std::uint64_t>() &&
                        payload.data.size() == original.at("bytes").get<std::uint64_t>(),
                    descriptor.name + ": not the exact original packed payload range");
            const auto elements = descriptor.shape[0] * descriptor.shape[1];
            // All registered shapes have aligned code planes; stored E2M1/E4M3 words are
            // followed by one FP32 weight divisor. Read it directly from the original file.
            const auto expected_weight = scalar(original.at("offset").get<std::uint64_t>() +
                                                elements / 2 + elements / 16);
            const auto expected_input = scalar(stored.at(divisor).at("offset").get<std::uint64_t>());
            require(weight.weight_scale_divisor_bits == expected_weight &&
                        weight.input_scale_divisor_bits == expected_input,
                    descriptor.name + ": scale pairing differs from source FP32 words");
            ++projections;
        };
        for (std::size_t layer = 0; layer < plan.bindings.text_layers.size(); ++layer) {
            const auto& weights = plan.bindings.text_layers[layer];
            const std::string prefix = "text/layers/" + std::to_string(layer) + "/";
            if (weights.is_full_attention) {
                check(std::get<FusedAttentionProjectionPlan>(weights.attention.projection)
                          .query_key_gate_value,
                      prefix + "attention/input_projection/input_scale_divisor");
                check(weights.attention.output,
                      prefix + "attention/output_projection/input_scale_divisor");
            } else {
                check(std::get<FusedGdnInputProjectionPlan>(weights.gdn.input_projection)
                          .query_key_value_z,
                      prefix + "gdn/input_projection/input_scale_divisor");
                check(weights.gdn.output,
                      prefix + "gdn/output_projection/input_scale_divisor");
                require(std::get<FusedGdnControlProjectionPlan>(weights.gdn.control_projection)
                            .a_b_projection.format == NumericFormat::BF16,
                        "QUASAR GDN controls lost the artifact's BF16 boundary");
            }
            check(weights.mlp.gate_up, prefix + "mlp/gate_up_projection/input_scale_divisor");
            check(weights.mlp.down, prefix + "mlp/down_projection/input_scale_divisor");
        }
        require(projections == 256 &&
                    plan.bindings.token_embedding.format == NumericFormat::W8G32_F16S &&
                    plan.bindings.output_head.format == NumericFormat::W8G32_F16S &&
                    plan.bindings.mtp_format == NumericFormat::W8G32_F16S,
                "QUASAR storage profile is incomplete");
        int mtp_objects = 0;
        for (const auto& placement : plan.materialization.device_objects) {
            const auto name = object_name(binder.descriptor(placement.object));
            require(!name.starts_with("vision/"),
                    "text-only QUASAR uploaded a Vision object to a GPU");
            if (name.starts_with("mtp/")) { ++mtp_objects; }
        }
        require(backend == ninfer::SpeculativeBackend::Mtp ? mtp_objects > 0 : mtp_objects == 0,
                "MTP startup selection is incorrect");
    }

    // Vision startup: the file carries the official Vision encodings, so the shared Vision route
    // must admit them and place every Vision object on the primary device.
    {
        Binder binder(reader, 2);
        const auto plan = bind_artifact(binder, WeightsProfile::Qwen38QuasarNvfp4,
                                        {.vision = true, .speculative = ninfer::SpeculativeBackend::Mtp},
                                        2);
        int vision_objects = 0;
        for (const auto& placement : plan.materialization.device_objects) {
            const auto name = object_name(binder.descriptor(placement.object));
            if (name.starts_with("vision/")) {
                require(placement.device == 0,
                        std::string(name) + ": QUASAR Vision must stay on the primary device");
                ++vision_objects;
            }
        }
        require(vision_objects > 0, "QUASAR Vision startup placed no Vision objects");
    }
}
} // namespace

int main() {
    const char* path = std::getenv("NINFER_QUASAR_ARTIFACT");
    if (!path || !*path) { std::cout << "SKIP: set NINFER_QUASAR_ARTIFACT\n"; return 77; }
    try {
        verify(path);
        std::cout << "OK QUASAR exact packed ranges, 256 scale pairs, TP2 None/MTP/Vision "
                     "placement and text-only zero GPU Vision objects\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}