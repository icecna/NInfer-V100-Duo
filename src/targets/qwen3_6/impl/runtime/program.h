#pragma once
#include "targets/qwen3_6/impl/runtime/instance.h"
// Qwen3.6 family runtime implementation; instantiated only by exact variants.

#include "core/arena.h"
#include "core/gdn_replay_records.h"
#include "core/host_kv_arena.h"
#include "ninfer/ops/allreduce.h"
#include "ninfer/ops/sampling.h"
#include "core/decode_graph.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include "targets/qwen3_6/impl/runtime/host_prefix_table.h"
#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/dflash_context.h"
#include "targets/qwen3_6/impl/runtime/linear_state_slots.h"
#include "targets/qwen3_6/impl/runtime/prefix_identity.h"
#include "runtime/engine/context_cache/prefix_index.h"
#include "runtime/engine/context_cache/retention.h"
#include "targets/qwen3_6/impl/runtime/text_context.h"
#include "targets/qwen3_6/impl/runtime/vision_context.h"
#include "targets/qwen3_6/impl/runtime/vision_prefill.h"

#include <cstdint>
#include <array>
#include <memory>
#include <optional>
#include <span>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using PreparedPromptData    = qwen3_6::PreparedPromptData;
using RewriteCheckpointKind = qwen3_6::RewriteCheckpointKind;
using RewriteCheckpointSpec = qwen3_6::RewriteCheckpointSpec;

using ReusePath = ninfer::PrefixReusePath;

[[nodiscard]] constexpr bool is_rewrite_checkpoint_restore(ReusePath path) noexcept {
    return path == ReusePath::RestoreTurnCheckpoint || path == ReusePath::RestoreResponseCheckpoint;
}

[[nodiscard]] constexpr ReusePath restore_path(RewriteCheckpointKind kind) noexcept {
    return kind == RewriteCheckpointKind::TurnClosure ? ReusePath::RestoreTurnCheckpoint
                                                      : ReusePath::RestoreResponseCheckpoint;
}

enum class RewriteCheckpointAction : std::uint8_t {
    Drop,
    KeepExisting,
    ReclassifyExisting,
    CaptureNew,
    DeferCapture,
};

enum class MtpBridgeMode : std::uint8_t {
    None,
    BeforeSuffix,
    AfterExactHit,
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
struct RequestBasePlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
    std::shared_ptr<const qwen3_6::VisionControl> vision_control;
    std::size_t vision_transient_bytes = 0;
    std::optional<qwen3_6::RewriteCheckpointSpec> rewrite_checkpoint;
    bool allow_prefix_reuse = false;
    // Lane-independent content match against the host prefix table, found at base-plan time and
    // bound to the chosen lane by plan_request_for_lane. Versioned against the table so the
    // executor can re-plan after the entry was inserted or evicted in the meantime.
    std::optional<qwen3_6::detail::HostPrefixId> host_entry = std::nullopt;
    std::uint32_t host_reuse_base = 0;
    std::uint64_t host_table_version = 0;
    // Rolling per-frontier content digest and the program-fixed identity tag; together they form
    // the shortlist key an engine-level prefix index compares before the exact match runs.
    std::vector<PrefixDigest> prefix_digests;
    std::uint32_t prefix_identity_tag = 0;

    [[nodiscard]] PrefixShortlistKey prefix_shortlist_key(std::uint32_t frontier) const {
        if (frontier == 0 || frontier >= prefix_digests.size()) { return {}; }
        return PrefixShortlistKey{prefix_digests[frontier], frontier, prefix_identity_tag};
    }
};

template <>
struct RequestPlanImpl<NINFER_QWEN36_VARIANT> {
    runtime::RequestPlanSummary summary;
    NINFER_QWEN36_RUNTIME_NS::ReusePath reuse = NINFER_QWEN36_RUNTIME_NS::ReusePath::FullReset;
    std::uint32_t reuse_base                  = 0;
    // Host table entry this plan claims (set when the reuse comes from the content-addressed
    // table instead of the lane's own device-resident prefix). start_prefill_lane restores it
    // into the lane before the reuse path runs.
    std::optional<qwen3_6::detail::HostPrefixId> host_entry = std::nullopt;
    NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode mtp_bridge =
        NINFER_QWEN36_RUNTIME_NS::MtpBridgeMode::None;
    bool prepare_mtp = false;
    std::optional<NINFER_QWEN36_RUNTIME_NS::VisionPrefillPlan> vision;
    NINFER_QWEN36_RUNTIME_NS::RewriteCheckpointAction rewrite_checkpoint_action =
        NINFER_QWEN36_RUNTIME_NS::RewriteCheckpointAction::Drop;
    std::optional<qwen3_6::RewriteCheckpointSpec> rewrite_checkpoint_capture;
    ops::SamplingConfig sampling;
    std::uint32_t text_kv_page_entitlement    = 0;
    std::uint32_t backend_kv_page_entitlement = 0;
};

} // namespace ninfer::targets::qwen3_6::detail

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

using RequestPlanImpl     = qwen3_6::detail::RequestPlanImpl<Variant>;
using RequestBasePlanImpl = qwen3_6::detail::RequestBasePlanImpl<Variant>;

enum class PendingKind : std::uint8_t {
    None,
    Begin,
    Ordinary,
    Speculative,
};

struct PendingCandidate {
    PendingKind kind            = PendingKind::None;
    std::uint32_t base_E        = 0;
    std::uint32_t base_S        = 0;
    std::uint32_t prompt_tokens = 0;
    std::uint32_t produced      = 0;
};

enum class Lifecycle : std::uint8_t {
    Empty,
    Prefilling,
    Active,
    Pending,
    Complete,
};

struct RewriteCheckpoint {
    bool valid                 = false;
    RewriteCheckpointKind kind = RewriteCheckpointKind::TurnClosure;
    std::uint32_t frontier     = 0;
};

struct SequenceKVBundle {
    PagedKVAllocation text;
    std::optional<PagedKVAllocation> backend;
    // Rank 1's allocation in ITS OWN text KV pool at tp == 2. The two pools have identical page
    // geometry (only the per-page byte count differs, because rank 1 holds 2 of the 4 KV heads),
    // and every pool operation below is issued on both in the same order, so the two allocations
    // hold the same page ids and publish identical block tables.
    std::optional<PagedKVAllocation> text_peer;
    // Rank 1's allocation in ITS OWN MTP (backend) KV pool at tp == 2. Same lockstep argument as
    // `text_peer`: identical page geometry, every pool operation issued on both in the same order.
    std::optional<PagedKVAllocation> backend_peer;
};

// Physical home of a retained lane's paged KV. `Device` is the retain-skip case (the prefix did
// not fit the host arena, or host backing is disabled): pages stay in the pool until the lane
// is evicted, and reuse is lane-affine. `HostTransferIn` is the in-flight window between the D2H
// enqueue and its completion, during which the device pages are still held and the lane behaves
// exactly like `Device`. When the transfer completes the entry is inserted into the content-
// addressed table and the lane is released entirely (no `Host` residency state exists).
enum class RetainedResidency : std::uint8_t {
    None,
    Device,
    HostTransferIn,
};

// The completed host copies of a retained prefix (a content-addressed table entry), one arena
// allocation per device
// pool. Page counts live in the allocations themselves. Empty unless `Host`. The handles are
// shared so an optional shorter residency at an earlier frontier can reference the same whole
// pages as its terminal instead of copying them; the arena charges each allocation once.
struct HostKVResidency {
    std::shared_ptr<HostKVAllocation> text_local;
    std::shared_ptr<HostKVAllocation> text_peer;
    std::shared_ptr<HostKVAllocation> backend_local;
    std::shared_ptr<HostKVAllocation> backend_peer;
};

// Production payload of the content-addressed host prefix table (P2): a self-contained snapshot
// of one completed prefix. The entry owns every arena extent and no lane state references it,
// so ANY idle lane can claim it (cross-lane reuse). Content is the identity: the full frontier
// tokens plus the resident prefix identity the planner matches, and the bit-exact state copies
// the claim path restores:
//   - paged KV (text pool + MTP backend pool, per device);
//   - the GDN linear-attention CURRENT slot (per device, linear_state_slot_bytes layout);
//   - the rank-0 tail hidden vector (MTP bridge / zero-suffix sampling input).
// Rewrite-checkpoint state is NOT snapshotted: a prompt that only reaches the checkpoint
// frontier (the history diverged after it) misses and recomputes cold.
struct HostPrefixResidency {
    using Prompt = PreparedPromptData;
    HostKVResidency kv;
    std::vector<TokenId> tokens;
    qwen3_6::detail::ResidentPrefixIdentity identity;
    std::optional<HostKVAllocation> gdn_local;
    std::optional<HostKVAllocation> gdn_peer;
    std::optional<HostKVAllocation> tail_local;
    std::size_t tail_bytes     = 0;
    std::uint32_t origin_lane  = 0; // observability only; claims are lane-free
    std::size_t reuse_boundary = 0;
};

struct DecodeGraphProfile {
    std::uint32_t batch_size             = 1;
    std::uint32_t min_execution_frontier = 0;
    std::uint32_t max_execution_frontier = 0;
    std::uint32_t topology_class         = 0;
    DecodeGraphDefinition definition;
};

struct DecodeGraphTopology {
    std::uint32_t topology_class = 0;
    DecodeGraphExecutable executable;
    std::optional<std::size_t> installed_profile;
};

struct DecodeGraphFamily {
    std::vector<DecodeGraphProfile> profiles;
    std::vector<DecodeGraphTopology> topologies;
};

// Target model continuation for one logical sequence. This state remains meaningful after the
// request which produced it has finished, so it is deliberately separate from request lifecycle,
// output, sampling, and round-control state.
struct SequenceState {
    std::optional<SequenceKVBundle> kv;
    Tensor tail_hidden;
    Tensor rewrite_checkpoint_hidden;
    std::uint32_t lane = 0;

    std::uint32_t execution_frontier = 0;
    std::uint32_t ledger_frontier    = 0;
    std::vector<TokenId> ledger;
    qwen3_6::detail::ResidentPrefixIdentity prefix_identity;
    std::int32_t rope_delta               = 0;
    std::uint32_t text_kv_valid           = 0;
    std::uint32_t mtp_kv_valid            = 0;
    std::uint32_t dflash_context_frontier = 0;
    std::array<TokenId, qwen3_6::kMtpDecodeMaximumDrafts> mtp_drafts{};
    std::uint32_t mtp_draft_count = 0;
    bool tail_hidden_valid        = false;
    bool retained                 = false;
    RetainedResidency retained_residency = RetainedResidency::None;
    RewriteCheckpoint rewrite_checkpoint;
    // Session lineage of the request that owns this lane; published with the retained entry so a
    // later request from the same lineage can reuse it, and anonymous requests still can.
    std::string session_key;
};

// Request/round control is not retained with a reusable SequenceState. A later concurrent Engine
// gives every occupied request slot its own instance of this state.
struct RequestControl {
    Lifecycle lifecycle = Lifecycle::Empty;
    PendingCandidate pending;
    ops::SamplingConfig sampling_host;
    GenerationTimings timings;
    SpeculativeStats speculative_stats;

    struct Prefill {
        PreparedPromptData prompt;
        std::optional<VisionPrefillPlan> vision_plan;
        std::unique_ptr<schedule::VisionPrefillSession> vision;
        runtime::TransientRegion transient;
        std::optional<RewriteCheckpointSpec> rewrite_checkpoint_capture;
        std::uint32_t base               = 0;
        std::uint32_t cursor             = 0;
        std::uint32_t prompt_tokens      = 0;
        std::uint32_t initial_mtp_extent = 0;
        double elapsed_seconds           = 0.0;
        bool prepare_mtp                 = false;
        ReusePath reuse                  = ReusePath::FullReset;
        MtpBridgeMode mtp_bridge         = MtpBridgeMode::None;
    };

    std::optional<Prefill> prefill;
};

// Rank 1's complete runtime mirror: its own arenas, its own shard of the weights, its own halved
// decoder state, and its own RoundState. It owns no bookkeeping -- lanes, page accounting,
// sampling and the pinned host round buffers all live once, on rank 0.
struct PeerRuntime {
    PeerRuntime(DeviceContext& peer_device, const LoadedModelData& peer_model,
                const SequencePlanImpl& plan);

    PeerRuntime(const PeerRuntime&)            = delete;
    PeerRuntime& operator=(const PeerRuntime&) = delete;

    DeviceContext& device;
    const LoadedModelData& model;
    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    std::unique_ptr<qwen3_6::DecoderState> decoder;
    // Rank 1's own GDN replay records: the speculative verify round records this device's own
    // head/channel shard and folds it here, so the two devices commit the same accepted prefix
    // from records neither ever exchanges.
    std::optional<GdnReplayRecords> replay_records;
    qwen3_6::RoundState io;
    Tensor prefill_hidden;
    // Rank 1's OWN penalty counters. `ops::SamplingConfig::token_counts` is a raw device pointer,
    // and the MTP round's acceptance runs on both devices (see the replicated-accept note in
    // mtp_impl.h), so rank 1 must never be handed rank 0's. It is not a cache: rank 0 remains the
    // source of truth (ProgramImplCore::install_sampling zeroes both, and the one increment that
    // happens on rank 0 alone -- prefill's bonus token -- is copied across before the first decode
    // round), and thereafter both are advanced by the same Op over bit-identical inputs.
    Tensor token_counts;
};

class ProgramImplCore {
public:
    ProgramImplCore(const LoadedModelData& model, const LoadedModelData* peer_model,
                    const SequencePlanImpl& plan, ExecutionContext& execution);
    ~ProgramImplCore() noexcept;

    [[nodiscard]] RequestBasePlan
    plan_request_base(const PreparedPromptData& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan plan_request_for_lane(std::uint32_t lane,
                                                    const PreparedPromptData& prompt,
                                                    const RequestBasePlan& base);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane, const RequestPlan& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane, const RequestPlan& plan,
                                           bool evict_self = false) const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                PreparedPromptData&& prompt,
                                                                RequestPlan&& plan,
                                                                runtime::TransientRegion transient);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill_lane(std::uint32_t lane);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_batch(std::span<const std::uint32_t> lanes,
                 std::span<const runtime::RoundBudget> budgets);
    void resolve_prefill_lane(std::uint32_t lane, bool terminal);
    void resolve_pending_batch(std::span<const std::uint32_t> lanes,
                               std::span<const std::uint32_t> accepted_tokens,
                               std::span<const std::uint8_t> terminal,
                               std::span<const std::uint8_t> cancelled);
    void abort_lane(std::uint32_t lane) noexcept;
    [[nodiscard]] std::uint64_t lane_release_generation(std::uint32_t lane) const noexcept;
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    void evict_retained_lane(std::uint32_t lane) noexcept;
    // Polls in-flight retain D2H transfers; a completed transfer releases its device pages and
    // publishes the entry into the content-addressed table (releasing the lane entirely), or,
    // when the lane moved on first, just drops the host extents. Called at every executor
    // boundary; the executor re-plans queued requests against the new host_table_version().
    void harvest_host_retains();
    // Blocks until the given lane's in-flight retain D2H (if any) is finalized.
    void wait_host_retain(std::uint32_t lane);
    // Blocks until every in-flight retain D2H is finalized. Admission planning calls this
    // before reading the table so a just-finished conversation is never missed by the content
    // pass and an in-flight prefix is never destroyed mid-transfer.
    void wait_all_host_retains();
    // Monotonic table mutation counter (see `host_table_version_`).
    [[nodiscard]] std::uint64_t host_table_version() const noexcept {
        return host_table_version_;
    }
    [[nodiscard]] bool has_host_retain_in_flight() const noexcept;
    [[nodiscard]] std::size_t host_kv_cache_capacity() const noexcept;
    [[nodiscard]] std::size_t host_kv_cache_used_bytes() const noexcept;
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;

    void reset_memory_peaks() noexcept;

    const LoadedModelData& model;
    ExecutionContext& execution;
    DeviceContext& device;
    const int tp;
    const std::uint32_t capacity;
    const std::uint32_t kv_capacity;
    const std::uint32_t max_concurrency;
    const std::uint32_t prefill_chunk;
    const std::uint32_t draft_window;
    const SpeculativeBackend speculative_backend;
    const DType kv_dtype;
    const std::int32_t kv_quant_group;
    const ProposalHead proposal_head;
    const bool vision_enabled;
    // Program-fixed identity tag mixed into every shortlist key this program publishes.
    const std::uint32_t prefix_identity_tag;
    const bool use_cuda_graph;
    const std::size_t kv_payload_bytes;
    const std::size_t gdn_state_bytes;
    const std::size_t graph_allowance_bytes;
    // Measured graph residency per device (index = rank). At tp1 only [0] is populated. At tp2 the
    // single cross-device graph materializes driver state on BOTH devices, and each is checked
    // against the SAME per-device allowance -- graph_allowance_bytes is a per-device budget, like
    // every other field in device_reservation_bytes.
    std::array<std::size_t, 2> graph_observed_bytes{0, 0};
    // Node count of ONE captured decode graph (the first profile of the captured family). At tp2
    // one graph holds both devices' nodes, so this is the direct measurement of whether the peer's
    // half of the schedule was captured rather than left out.
    std::size_t graph_node_count = 0;
    const WorkspacePlan workspace_plan;

    DeviceArena persistent;
    DeviceArena workspace_storage;
    WorkspaceArena work;
    // YaRN rotary state, resolved once at construction.
    //
    // `rope_frequency_storage[rank]` is a dedicated 32-float device allocation on rank `rank`'s OWN
    // device, made once here and never touched again: CUDA Graph capture bakes the pointer into the
    // replayed rope launch node, so it must outlive every replay, and at tp 2 each rank ropes its
    // own head-local q/k on its own device, where a pointer into the other rank's allocation is not
    // addressable. It is deliberately NOT part of the planned persistent arena: under
    // `RopeMode::Native` nothing is allocated at all, which is what keeps a native plan and its
    // memory summary byte-identical to the pre-YaRN engine.
    //
    // `rope_frequency[rank]` is the descriptor every text rope call site reads (through
    // `ExecutionCore::rope_frequency` -> `TextContext`); a null `inv_frequency` IS the native path.
    std::array<DeviceBuffer, 2> rope_frequency_storage;
    std::array<ops::RopeFrequencyOverride, 2> rope_frequency{};
    const RopeMode rope_mode;
    const std::uint32_t effective_max_context;
    const double yarn_mscale;
    std::optional<PeerRuntime> peer;
    std::optional<ops::PeerEvents> peer_events;
    // Created once at tp2 when graphs are on; forks rank 1's stream into rank 0's capture.
    std::optional<DecodeGraphPeerBridge> graph_bridge;
    std::optional<schedule::TpPeerCore> peer_core;
    std::unique_ptr<qwen3_6::DecoderState> decoder;
    std::optional<GdnReplayRecords> replay_records;
    std::optional<DFlashPersistentState> dflash;
    qwen3_6::RoundState io;
    Tensor prefill_hidden;
    Tensor sampling_config;
    Tensor token_counts;
    Tensor tail_hidden_store;
    Tensor rewrite_checkpoint_hidden_store;

    // Host-backed retained prefixes. `host_kv_cache_bytes` is the TOTAL across devices; each
    // arena holds its own device's share. Zero disables the whole feature and every host path
    // below is a no-op (retains stay device-resident exactly as before). Destruction is in
    // reverse declaration order, so every member that owns HostKVAllocations -- the prefix
    // table, `sequences`, and the transfer table -- MUST be declared AFTER the arenas: an
    // allocation released from a destroyed arena writes into freed free-extent/descriptor
    // vectors and corrupts the heap.
    std::size_t host_kv_cache_bytes = 0;
    std::optional<HostKVArena> host_kv_arena;
    std::optional<HostKVArena> host_kv_arena_peer;
    qwen3_6::detail::HostPrefixTable<HostPrefixResidency> host_prefixes;
    // Shortlist index over the hosted residencies: keyed by (identity_tag, frontier, digest) so a
    // request compares digests before the authoritative token/media match. Kept in lockstep with
    // host_prefixes on insert, evict, and clear.
    runtime::PrefixIndex<qwen3_6::detail::HostPrefixId, qwen3_6::detail::PrefixShortlistKey>
        host_prefix_index;
    // Retention authority for the hosted residencies: owner/session lineage, demand, and the
    // eviction order. The table owns storage; this owns policy.
    runtime::ContextRetentionPolicy host_retention;
    std::uint64_t host_prefix_id_counter = 0;

    std::array<SequenceState, kMaximumConcurrency> sequences;
    std::array<RequestControl, kMaximumConcurrency> requests;
    // Per-lane completion events for in-flight D2H transfers: [lane][0] rank 0 stream,
    // [lane][1] rank 1 stream (tp2 only). Created once when the feature is enabled.
    std::array<std::array<cudaEvent_t, 2>, kMaximumConcurrency> host_retain_events{};
    // Dedicated pinned-transfer streams: the retain D2H must not stall the compute stream, on
    // which the other lanes' decode rounds run; the gate event serializes the copy behind the
    // terminal round's own work (and the KV pages it just wrote) before touching them.
    cudaStream_t host_copy_stream     = nullptr;
    cudaEvent_t host_copy_gate        = nullptr;
    cudaStream_t host_copy_stream_peer = nullptr;
    cudaEvent_t host_copy_gate_peer   = nullptr;
    // One in-flight retain D2H transfer per lane. The host extents are owned here until both
    // device events report completion, so an abandoned transfer (its lane was cleared or reused
    // first) still lands its copies in memory that nobody else has been handed yet, and is only
    // dropped back to the arena once the reads/writes have retired.
    struct HostRetainTransfer {
        std::optional<HostKVAllocation> text_local;
        std::optional<HostKVAllocation> text_peer;
        std::optional<HostKVAllocation> backend_local;
        std::optional<HostKVAllocation> backend_peer;
        std::optional<HostKVAllocation> gdn_local;
        std::optional<HostKVAllocation> gdn_peer;
        std::optional<HostKVAllocation> tail_local;
        // Optional second residency at the rewrite-checkpoint frontier: it shares the terminal's
        // KV pages and owns only the GDN state and hidden the rewrite checkpoint froze there.
        // Publishing it lets a later request that re-renders only up to that boundary reuse the
        // prefill state instead of recomputing the whole prompt.
        std::optional<HostKVAllocation> prefix_gdn_local;
        std::optional<HostKVAllocation> prefix_gdn_peer;
        std::optional<HostKVAllocation> prefix_tail_local;
        std::uint32_t prefix_frontier = 0;
        bool has_prefix               = false;
        bool abandoned = false;
    };
    std::array<std::optional<HostRetainTransfer>, kMaximumConcurrency> host_retain_transfers{};
    // Per-lane monotonically increasing generation, bumped whenever a lane's device state
    // changes in a way that invalidates cached lane plans: clearing a lane (any eviction,
    // completion) and a host-retain transfer retiring (the lane goes device-retained ->
    // empty). The executor stamps this generation into its cached lane plans instead of
    // tracking the events itself.
    std::array<std::uint64_t, kMaximumConcurrency> lane_release_generation_{};
    std::uint64_t host_lru_counter = 0;
    // Monotonic demand stamp: the last request that reused a host entry resists eviction.
    std::uint64_t host_demand_counter = 0;
    // Bumped on every table mutation (insert, evict, claim-consume). Plans stamp the version at
    // computation time; a mismatch means a referenced entry may have appeared or disappeared and
    // the request must re-plan. The version, not lane ids, is the invalidation unit: host
    // entries no longer occupy lanes.
    std::uint64_t host_table_version_ = 0;

    DecodeGraphFamily ordinary_graphs;
    DecodeGraphFamily mtp_graphs;
    DecodeGraphFamily dflash_graphs;

    PinnedHostBuffer round_host;
    TokenId* host_tokens = nullptr;
    // Debug-only capture of the round's full-vocabulary logits, raw BF16 bits. OFF by default and
    // completely inert until enable_logits_capture(true) allocates the host buffer: with it off,
    // copy_round_logits() is a predictable branch and nothing is allocated or copied, so no
    // production path (tp1 or tp2) is changed. With it on, the buffer is overwritten every time a
    // token is sampled from io.logits at prefill finalization, for both tp1 and tp2 (io is rank
    // 0's window; logits_tp2 gathers into it the same way the tp1 leaf writes it), so no
    // tp-specific code is needed. Prefill only -- the decode path is CUDA-graph-capturable and an
    // unconditional memcpy inside a captured region is rejected at capture time; a probe that
    // requests exactly one output token completes on prefill's own sampled token and never runs a
    // decode round, which is what the parity harness (tools/tp2/parity.cpp) uses. Valid only
    // immediately after Engine::generate()/wait() returns to the calling thread for a single-lane
    // engine -- concurrent lanes would overwrite it out of order.
    std::vector<std::uint16_t> logits_capture;
    bool logits_capture_enabled = false;
    std::optional<PinnedHostBuffer> ordinary_host;
    qwen3_6::OrdinaryDecodeIngress* ordinary_host_ingress = nullptr;
    qwen3_6::OrdinaryDecodeEgress* ordinary_host_egress   = nullptr;
    // Rank 1's own pinned ordinary ingress: rank 0's record with every row's
    // `sampling[row].token_counts` nulled. Rank 1 mirrors the ordinary round for its half of the
    // weights and never samples (the output head is vocabulary-split and sampling belongs to rank
    // 0), so the sampling configs in its frame are inert -- but they must not carry a rank-0
    // DEVICE address into rank 1's frame: dereferencing one from rank 1 is a silent cross-device
    // fault. A separate pinned buffer rather than a patched copy at issue time, for the same
    // reason as `mtp_peer_host_ingress`: the upload is inside the captured decode graph, which
    // re-reads this exact host address at every replay.
    std::optional<PinnedHostBuffer> ordinary_peer_host;
    qwen3_6::OrdinaryDecodeIngress* ordinary_peer_host_ingress = nullptr;
    bool peer_egress_check_enabled     = false;
    std::uint64_t peer_egress_rounds     = 0;
    std::uint64_t peer_egress_mismatches = 0;
    std::optional<PinnedHostBuffer> mtp_host;
    qwen3_6::MtpDecodeIngress* mtp_host_ingress = nullptr;
    qwen3_6::MtpDecodeEgress* mtp_host_egress   = nullptr;
    // Rank 1's own pinned MTP ingress: byte-for-byte rank 0's record except that each row's
    // `sampling[row].token_counts` names rank 1's counter lane. It has to be a separate pinned
    // buffer rather than a patched copy made at issue time, because the ingress upload is inside
    // the captured decode graph and the graph re-reads this exact host address at every replay.
    std::optional<PinnedHostBuffer> mtp_peer_host;
    qwen3_6::MtpDecodeIngress* mtp_peer_host_ingress = nullptr;
    std::optional<PinnedHostBuffer> dflash_host;
    qwen3_6::DFlashDecodeIngress* dflash_host_ingress = nullptr;
    qwen3_6::DFlashDecodeEgress* dflash_host_egress   = nullptr;

    std::size_t workspace_logical_peak_bytes = 0;

    // See logits_capture's comment. Read-only; the caller owns thread-safety (single-lane, read
    // after wait() returns). Empty while capture is disabled.
    [[nodiscard]] std::span<const std::uint16_t> last_round_logits_bf16() const noexcept {
        return logits_capture;
    }

    // Turns the debug capture above on or off. Enabling sizes the host buffer to the round's
    // logits row count; disabling releases it. Must not be called while a round is in flight (the
    // executor holds its execution mutex across this call).
    void enable_logits_capture(bool enabled);

    // Debug-only, OFF by default: after each MTP decode round at tp == 2, read rank 1's MTP egress
    // back and compare it field for field with rank 0's. The two ranks run the acceptance Op over
    // bit-identical inputs, so their egress records are argued to agree; enabling this turns that
    // induction into a measurement. Costs one ~1 KiB device-to-host copy plus a host compare per
    // round while enabled, and nothing at all while off. Counters are cumulative over the
    // Program's lifetime; a mismatch is counted (per row, per field) rather than thrown, so a test
    // can read the totals after a clean run.
    void enable_peer_egress_check(bool enabled) noexcept;
    [[nodiscard]] std::uint64_t peer_egress_check_rounds() const noexcept {
        return peer_egress_rounds;
    }
    [[nodiscard]] std::uint64_t peer_egress_check_mismatches() const noexcept {
        return peer_egress_mismatches;
    }

private:
    void clear_lane(SequenceState& sequence, RequestControl& request) noexcept;
    void ordered_reset(SequenceState& sequence);
    void prepare_graphs();
    // Copies rank 0's counter lane onto rank 1. Called once, after prefill's bonus token is
    // sampled -- the only counter increment that happens on rank 0 and not on rank 1.
    [[nodiscard]] static Tensor token_counts_lane(const Tensor& storage, std::uint32_t lane);
    void publish_peer_token_counts(const SequenceState& sequence);
    // Mirrors `mtp_host_ingress` into `mtp_peer_host_ingress`, swapping every row's counter
    // pointer for rank 1's. No-op at tp1 or without MTP.
    void publish_peer_mtp_ingress(std::span<const std::uint32_t> lanes);
    // Mirrors `ordinary_host_ingress` into `ordinary_peer_host_ingress` with every row's counter
    // pointer nulled. No-op at tp1 or without an ordinary frame.
    void publish_peer_ordinary_ingress();
    // Debug-only: reads rank 1's MTP egress back and compares it, field for field, with rank 0's.
    // No-op unless the check is enabled and a peer exists.
    void check_peer_mtp_egress(std::size_t rows);
    void install_sampling(SequenceState& sequence, RequestControl& request,
                          const ops::SamplingConfig& config);
    void set_device_i32(Tensor& tensor, std::int32_t value);
    void set_peer_i32(Tensor& tensor, std::int32_t value);
    void copy_tail(SequenceState& sequence, const Tensor& source);
    void copy_round_token();
    void copy_round_logits();
    void resolve_non_speculative_pending(SequenceState& sequence, RequestControl& request,
                                         std::uint32_t accepted_tokens, bool terminal);
    [[nodiscard]] runtime::PrefillStepResult advance_prefill(SequenceState& sequence,
                                                             RequestControl& request);
    void enqueue_dflash_context_append(std::span<const std::uint32_t> lanes,
                                       std::span<const std::uint32_t> starts,
                                       std::span<const std::uint32_t> counts);
    void validate_licensed_tokens(std::span<const TokenId> tokens) const;
    void mark_workspace_usage(std::size_t phase_bytes) noexcept;
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_ordinary_batch(std::span<const std::uint32_t> lanes,
                          std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_mtp_batch(std::span<const std::uint32_t> lanes,
                     std::span<const runtime::RoundBudget> budgets);
    [[nodiscard]] runtime::BatchedGeneratedRound
    decode_dflash_batch(std::span<const std::uint32_t> lanes,
                        std::span<const runtime::RoundBudget> budgets);
    void reserve_sequence_kv(SequenceState& sequence, std::uint32_t text_pages,
                             std::uint32_t backend_pages);
    void resize_sequence_kv_entitlement(SequenceState& sequence, std::uint32_t text_pages,
                                        std::uint32_t backend_pages);
    void bind_sequence_kv(SequenceState& sequence);
    void unbind_sequence_kv(SequenceState& sequence) noexcept;
    // Host-backing of retained prefixes (no-op with host_kv_cache_bytes == 0).
    void init_host_kv_cache(const SequencePlanImpl& plan);
    void retain_host_sequence(SequenceState& sequence, std::uint32_t lane);
    void claim_host_sequence(SequenceState& sequence, const RequestPlanImpl& plan, std::uint32_t base);
    void evict_host_entry(qwen3_6::detail::HostPrefixId id) noexcept;
    void complete_host_transfer(std::uint32_t lane, bool abandoned);
    void copy_kv_pages_to_host(const PagedKVPool& pool, const PagedKVAllocation& allocation,
                               std::uint32_t pages, const HostKVAllocationView& host,
                               cudaStream_t stream) const;
    void copy_kv_pages_from_host(const PagedKVPool& pool, const PagedKVAllocation& allocation,
                                 std::uint32_t pages, const HostKVAllocationView& host,
                                 cudaStream_t stream) const;
    bool host_transfer_events_done(std::uint32_t lane) const;
    bool host_retain_copies_done() const;
    void host_transfer_drain(std::uint32_t lane) const;
    void materialize_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                                 std::uint32_t backend_tokens = 0);
    void trim_sequence_kv(SequenceState& sequence, std::uint32_t main_tokens,
                          std::uint32_t backend_tokens = 0);
    void release_sequence_growth_entitlement(SequenceState& sequence) noexcept;
    [[nodiscard]] qwen3_6::PagedKVCache* backend_kv_cache() noexcept;
    [[nodiscard]] const qwen3_6::PagedKVCache* backend_kv_cache() const noexcept;
    [[nodiscard]] std::uint32_t backend_kv_valid(const SequenceState& sequence) const noexcept;
    [[nodiscard]] qwen3_6::PagedKVCacheView text_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_6::PagedKVCacheView mtp_kv_view(const SequenceState& sequence) const;
    [[nodiscard]] qwen3_6::PagedKVCacheView mtp_kv_view_peer(const SequenceState& sequence) const;
};

} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS

namespace ninfer::targets::qwen3_6::detail {

template <>
class ProgramImpl<NINFER_QWEN36_VARIANT> final : public NINFER_QWEN36_RUNTIME_NS::ProgramImplCore {
public:
    using NINFER_QWEN36_RUNTIME_NS::ProgramImplCore::ProgramImplCore;
};

} // namespace ninfer::targets::qwen3_6::detail
