// Opt-in TP2 host KV cache round-trip gate on the real artifact.
//
// The host retained prefix layer (EngineOptions::context_host_cache_bytes > 0) offloads a
// completed lane's paged KV to per-device host RAM at the terminal commit (D2H on a dedicated
// copy stream) and restores it at admission (synchronous H2D claim). The behavioral oracle is
// exact token identity against a control run on the same artifact with the host cache disabled:
// both copies are byte-exact, so a host-resident reuse must produce the same greedy output,
// frontier, and MTP statistics as a device-resident reuse. The memory summary proves the
// offload actually happened (non-zero host occupancy on the host run, zero on the control).
//
// NINFER_V100_DUO_ARTIFACT=/path/to/model.ninfer ctest -R v100-duo_host_kv_real

#include "ninfer/engine.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr std::uint32_t kCapacity = 8192;
constexpr std::uint32_t kOutput   = 32;
constexpr std::size_t kHostCache  = 256ULL << 20;
using Tokens = std::vector<ninfer::TokenId>;

void require(bool condition, const std::string& message) {
    if (!condition) { throw std::runtime_error(message); }
}

ninfer::EngineOptions engine_options(const char* artifact, bool host_cache) {
    ninfer::EngineOptions options;
    options.artifact_path              = artifact;
    options.tp                         = 2;
    options.devices                    = {0, 1};
    options.max_context                = kCapacity;
    options.kv_capacity                = ninfer::KvCapacityPolicy::explicit_capacity(kCapacity);
    options.prefill_chunk              = 256;
    options.kv_cache                   = ninfer::KvCacheStorage::Int8Group64;
    options.use_cuda_graph             = false;
    options.speculative.backend        = ninfer::SpeculativeBackend::Mtp;
    options.speculative.draft_tokens   = 3;
    options.speculative.proposal_head  = ninfer::ProposalHead::Optimized;
    options.context_host_cache_bytes   = host_cache ? kHostCache : 0;
    return options;
}

ninfer::RequestOptions request_options(bool reuse) {
    ninfer::RequestOptions options;
    options.execution.requested_output_tokens = kOutput;
    options.execution.allow_prefix_reuse      = reuse;
    options.execution.sampling.temperature    = 0.0F;
    options.execution.sampling.presence_penalty = 0.0F;
    options.execution.sampling.frequency_penalty = 0.0F;
    options.stop.include_model_defaults  = false;
    return options;
}

ninfer::ChatMessage message(ninfer::ChatRole role, std::string text) {
    ninfer::ChatMessage result;
    result.role = role;
    result.parts.push_back({ninfer::MessagePartKind::Text, std::move(text), {}});
    return result;
}

ninfer::PromptInput coding_prompt() {
    std::string text = "Review these Python functions. Then implement a bounded FIFO queue "
                       "with push, pop, and unit tests. Explain empty and full behavior.\n\n";
    for (int i = 0; i < 192; ++i) {
        text += "def add_" + std::to_string(i) + "(value):\n    return value + " +
                std::to_string(i) + "\n\n";
    }
    ninfer::PromptInput result;
    result.options.enable_thinking = false;
    // Retain the response boundary including the deterministic no-thinking prologue.
    result.options.preserve_thinking = true;
    result.messages.push_back(message(ninfer::ChatRole::User, std::move(text)));
    return result;
}

struct Round {
    std::string name;
    Tokens output;
    std::uint32_t reused = 0;
};

Round run(ninfer::Engine& engine, std::string name, ninfer::PreparedPrompt prompt, bool reuse) {
    auto result = engine.generate(std::move(prompt), request_options(reuse));
    require(result.generated_token_ids.size() == kOutput, name + ": unexpected output count");
    require(result.finish_reason == ninfer::FinishReason::OutputLimit,
            name + ": unexpected finish reason");
    require(result.speculative.enabled && result.speculative.rounds > 0,
            name + ": MTP3 did not execute");
    std::cout << name << " reused=" << result.reused_prompt_tokens << " ttft_s="
              << result.timings.first_token_seconds << std::endl;
    return {std::move(name), result.generated_token_ids, result.reused_prompt_tokens};
}

Tokens continuation(const Tokens& prompt, const Tokens& first_output, const Tokens& next_turn) {
    Tokens result = prompt;
    result.insert(result.end(), first_output.begin(), first_output.end());
    // Complete the current assistant response, then append a separately rendered user turn.
    result.push_back(248046);
    result.push_back(198);
    result.insert(result.end(), next_turn.begin(), next_turn.end());
    return result;
}

struct Session {
    Tokens base_tokens;
    Tokens continuation;
    Round cold;
    Round append;
    std::size_t host_capacity = 0;
    std::size_t host_occupied = 0;
};

Session session(const char* artifact, bool host_cache) {
    Session s;
    {
        ninfer::Engine engine(engine_options(artifact, host_cache));
        require(engine.load_summary().tp == 2 && engine.load_summary().model_id == "qwen3.8-27b",
                "this test requires the TP2 Qwen3.8 artifact");
        const auto input = coding_prompt();
        s.base_tokens = engine.prepare(input).debug_token_ids();
        require(s.base_tokens.size() >= 2048 && s.base_tokens.size() + 512 < kCapacity,
                "coding fixture must have at least 2k tokens and leave headroom");
        s.cold = run(engine, host_cache ? "host_cold" : "cold", engine.prepare(input), false);
        require(s.cold.reused == 0, s.cold.name + ": cold run reported a reuse");

        if (host_cache) {
            // The terminal D2H is asynchronous on the dedicated copy stream; the executor
            // harvests it at the next loop boundary, i.e. when the next request is admitted.
            // Wait DETERMINISTICALLY for the copy to retire: the memory summary reports
            // host_kv_transfer_in_flight until the harvest completes the transfer (which
            // requires both device events to be done, i.e. the D2H bytes have moved). Only
            // then is the append's admission guaranteed to hit the Host-resident lane and
            // be served by the H2D claim from the arena. (Spinning extra requests here
            // would be wrong: they admit on the retained lane and its FullReset destroys the
            // very prefix under test.)
            for (int spin = 0; spin < 3000; ++spin) {
                if (engine.memory_summary().host_kv_retain_complete) { break; }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            require(engine.memory_summary().host_kv_retain_complete,
                    "terminal D2H did not retire within 30 s");
        }

        ninfer::PromptInput followup;
        followup.options.enable_thinking = false;
        followup.options.preserve_thinking = true;
        followup.messages.push_back(message(ninfer::ChatRole::User,
            "Continue the implementation, and include a test for popping from an empty queue."));
        const auto next_turn = engine.prepare(std::move(followup)).debug_token_ids();
        s.continuation = continuation(s.base_tokens, s.cold.output, next_turn);
        s.append = run(engine, host_cache ? "host_append" : "append",
                       engine.prepare_tokens(s.continuation), true);
        // The committed frontier is the prompt plus all decoded tokens minus the trailing
        // hidden-state boundary token (same formula as the device-resident prefix gate); the
        // follow-up turn's tokens are not part of it.
        require(s.append.reused == s.base_tokens.size() + kOutput - 1,
                s.append.name + ": reused a different committed frontier");

        const auto summary = engine.memory_summary();
        s.host_capacity = summary.host_kv_cache_bytes;
        s.host_occupied = summary.host_kv_cache_occupied_bytes;
        std::cout << (host_cache ? "host" : "control")
                  << " host_cache_capacity=" << s.host_capacity
                  << " host_cache_occupied=" << s.host_occupied << std::endl;
    }
    return s;
}

}  // namespace

int main() {
    const char* artifact = std::getenv("NINFER_V100_DUO_ARTIFACT");
    if (artifact == nullptr || *artifact == '\0') {
        std::cout << "skip: NINFER_V100_DUO_ARTIFACT is not set\n";
        return 77;
    }
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices < 2) {
        std::cout << "skip: V100 Duo host KV gate requires two CUDA devices\n";
        return 77;
    }
    try {
        const auto control = session(artifact, /*host_cache=*/false);
        const auto host    = session(artifact, /*host_cache=*/true);

        require(control.host_capacity == 0, "control must not create a host arena");
        require(host.host_capacity == kHostCache, "host run did not size the arena");
        require(host.host_occupied > 0,
                "host run left the arena empty: the terminal D2H offload did not happen");
        require(control.cold.output == host.cold.output, "cold runs diverged (unrelated)");
        require(control.append.output == host.append.output,
                "host round-trip (D2H retain + H2D claim) changed the decoded output");
        require(control.append.reused == host.append.reused,
                "host run restored a different prefix frontier");
        std::cout << "V100 Duo host KV round-trip: exact outputs, frontiers, and occupancy PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "V100 Duo host KV round-trip: " << error.what() << '\n';
        return 1;
    }
}