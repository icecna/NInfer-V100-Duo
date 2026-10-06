#pragma once

#include "ninfer/types.h"
#include "runtime/contract/transient_region.h"
#include "runtime/contract/types.h"
#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>

namespace ninfer {
struct DeviceContext;
struct ExecutionContext;
}

namespace ninfer::targets::qwen3_6 {

enum class TextPhase {
    Prefill,
    Verify,
};

struct GraphExecutionProfile {
    std::uint32_t min            = 0;
    std::uint32_t max            = 0;
    std::uint32_t topology_class = 0;
};

namespace detail {
template <class Variant>
struct SequencePlanImpl;
template <class Variant>
struct SequencePlannerImpl;
template <class Variant>
struct RequestPlanImpl;
template <class Variant>
struct RequestBasePlanImpl;
template <class Variant>
class ProgramImpl;
} // namespace detail

template <class Variant>
class SequencePlanner;

// These are the complete family execution types. Exact packages bind them to a private Variant;
// target selection remains outside this layer and happens once in the closed Engine registry.
template <class Variant>
class SequencePlan {
public:
    SequencePlan(SequencePlan&&) noexcept;
    SequencePlan& operator=(SequencePlan&&) noexcept;
    ~SequencePlan();

    SequencePlan(const SequencePlan&)            = delete;
    SequencePlan& operator=(const SequencePlan&) = delete;

    [[nodiscard]] std::uint32_t capacity() const noexcept;
    [[nodiscard]] std::uint32_t kv_capacity() const noexcept;
    [[nodiscard]] std::uint32_t max_concurrency() const noexcept;
    [[nodiscard]] std::size_t device_reservation_bytes() const noexcept;
    [[nodiscard]] std::size_t workspace_capacity_bytes() const noexcept;
    [[nodiscard]] std::size_t request_transient_capacity_bytes() const noexcept;

public:
    // Family-private construction/storage seam; exact packages expose only the completed alias.
    explicit SequencePlan(std::unique_ptr<detail::SequencePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlanImpl<Variant>> impl_;

    template <class V>
    friend class SequencePlanner;
    template <class V>
    friend class detail::ProgramImpl;
};

template <class Variant>
class SequencePlanner {
public:
    SequencePlanner(SequencePlanner&&) noexcept;
    SequencePlanner& operator=(SequencePlanner&&) noexcept;
    ~SequencePlanner();

    SequencePlanner(const SequencePlanner&)            = delete;
    SequencePlanner& operator=(const SequencePlanner&) = delete;

    [[nodiscard]] const runtime::SequenceCapacityCurve& capacity_curve() const noexcept;
    [[nodiscard]] SequencePlan<Variant> finalize(std::uint32_t main_page_groups) &&;

public:
    explicit SequencePlanner(std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::SequencePlannerImpl<Variant>> impl_;

    template <class V>
    friend SequencePlanner<V> make_sequence_planner(DeviceContext&, const EngineOptions&,
                                                    typename V::WeightsProfile);
};

template <class Variant>
class RequestBasePlan {
public:
    RequestBasePlan(RequestBasePlan&&) noexcept;
    RequestBasePlan& operator=(RequestBasePlan&&) noexcept;
    ~RequestBasePlan();

    RequestBasePlan(const RequestBasePlan&)            = delete;
    RequestBasePlan& operator=(const RequestBasePlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    // Content-addressed host table stamp: the table version this plan was computed against.
    // A differing current version means the referenced entry may have changed; re-plan.
    [[nodiscard]] std::uint64_t host_table_version() const noexcept;

public:
    explicit RequestBasePlan(std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::RequestBasePlanImpl<Variant>> impl_;
};

template <class Variant>
class RequestPlan {
public:
    RequestPlan(RequestPlan&&) noexcept;
    RequestPlan& operator=(RequestPlan&&) noexcept;
    ~RequestPlan();

    RequestPlan(const RequestPlan&)            = delete;
    RequestPlan& operator=(const RequestPlan&) = delete;

    [[nodiscard]] const runtime::RequestPlanSummary& summary() const noexcept;
    // True when this plan's reuse comes from a content-addressed host table entry (the claim
    // materializes a fresh device bundle and therefore requires an EMPTY lane).
    [[nodiscard]] bool uses_host_entry() const noexcept;

public:
    // Family-private construction/storage seam. This header is repository-internal; exact
    // packages expose only the completed alias and never inspect this pointer.
    explicit RequestPlan(std::unique_ptr<detail::RequestPlanImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::RequestPlanImpl<Variant>> impl_;
};

template <class Variant>
class Program {
public:
    ~Program() noexcept;

    Program(const Program&)            = delete;
    Program& operator=(const Program&) = delete;
    Program(Program&&)                 = delete;
    Program& operator=(Program&&)      = delete;

    // Engine-internal fixed-lane execution surface. The public Engine owns scheduling; Program
    // owns target state images and executes one immutable decode batch membership.
    [[nodiscard]] RequestBasePlan<Variant>
    plan_request_base(const PreparedPrompt& prompt,
                      const runtime::ResolvedExecutionOptions& options);
    [[nodiscard]] RequestPlan<Variant> plan_request_for_lane(std::uint32_t lane,
                                                             const PreparedPrompt& prompt,
                                                             const RequestBasePlan<Variant>& base);
    [[nodiscard]] bool can_admit_lane(std::uint32_t lane,
                                      const RequestPlan<Variant>& plan) const noexcept;
    [[nodiscard]] bool
    can_admit_lane_after_retained_eviction(std::uint32_t lane,
                                           const RequestPlan<Variant>& plan,
                                           bool evict_self = false) const noexcept;
    [[nodiscard]] runtime::AdmissionResources admission_capacity() const noexcept;
    [[nodiscard]] runtime::PrefillStepResult start_prefill_lane(std::uint32_t lane,
                                                                PreparedPrompt&& prompt,
                                                                RequestPlan<Variant>&& plan,
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
    [[nodiscard]] bool has_retained_lane(std::uint32_t lane) const noexcept;
    void evict_retained_lane(std::uint32_t lane) noexcept;
    // Polls in-flight retain D2H transfers at an executor boundary: a completed transfer
    // publishes its self-contained entry into the content-addressed host table (releasing the
    // lane entirely and returning the device pages to the pool), or, when the lane moved on
    // first, just drops the host extents. No-op while the host cache is disabled. Callers
    // re-plan queued requests against the new host_table_version().
    void harvest_host_retains();
    // Blocks until the given lane's in-flight retain D2H (if any) is finalized.
    void wait_host_retain(std::uint32_t lane);
    // Blocks until every in-flight retain D2H is finalized. Admission planning calls it before
    // reading the table so a just-finished conversation is visible to the content pass instead
    // of being destroyed mid-transfer.
    void wait_all_host_retains();
    // Monotonic host-prefix-table mutation counter; a change invalidates plans stamped with an
    // older version (their referenced entry may have been inserted or evicted).
    [[nodiscard]] std::uint64_t host_table_version() const noexcept;
    // Monotonic per-lane generation, bumped when the lane's device state changes in a way that
    // invalidates cached lane plans (lane cleared, or a host-retain transfer retiring with the
    // lane going device-retained -> empty).
    [[nodiscard]] std::uint64_t lane_release_generation(std::uint32_t lane) const noexcept;
    [[nodiscard]] bool has_host_retain_in_flight() const noexcept;
    [[nodiscard]] GenerationTimings generation_timings_lane(std::uint32_t lane) const noexcept;
    [[nodiscard]] SpeculativeStats speculative_stats_lane(std::uint32_t lane) const noexcept;

    [[nodiscard]] MemorySummary memory_summary() const noexcept;
    void reset_memory_peaks() noexcept;

    // Debug-only, OFF by default: capture of the full-vocabulary logits behind the most recently
    // sampled token, as raw BF16 bits (see ProgramImplCore::logits_capture in the impl runtime).
    // Nothing is allocated, copied, or otherwise changed until enable_logits_capture(true) runs,
    // so an engine that never asks for it executes exactly as before. The span is valid only
    // immediately after the caller's own round completes on a single-lane engine, and is empty
    // while capture is disabled. Added for the greedy tp1-vs-tp2 parity harness
    // (tools/tp2/parity.cpp); not part of any wire-facing API.
    [[nodiscard]] std::span<const std::uint16_t> last_round_logits_bf16() const noexcept;
    void enable_logits_capture(bool enabled);

    // Debug-only, OFF by default: after each MTP decode round at tp == 2, compare rank 1's MTP
    // egress record with rank 0's. The ranks run the acceptance Op over bit-identical inputs, so
    // the records are argued to agree; enabling this measures it instead. Off, execution is
    // unchanged; on, it costs one small device-to-host copy and a host compare per round. Counters
    // are cumulative over the Program's lifetime. No-op at tp1 or without MTP.
    void enable_peer_egress_check(bool enabled) noexcept;
    [[nodiscard]] std::uint64_t peer_egress_check_rounds() const noexcept;
    [[nodiscard]] std::uint64_t peer_egress_check_mismatches() const noexcept;

private:
    explicit Program(std::unique_ptr<detail::ProgramImpl<Variant>> impl) noexcept;
    std::unique_ptr<detail::ProgramImpl<Variant>> impl_;

    template <class V>
    friend std::unique_ptr<Program<V>> create_program(const typename V::ModelView&,
                                                      const typename V::ModelView*,
                                                      typename V::WeightsProfile, SequencePlan<V>&&,
                                                      ExecutionContext&);
};

template <class Variant>
[[nodiscard]] SequencePlanner<Variant>
make_sequence_planner(DeviceContext& device, const EngineOptions& options,
                      typename Variant::WeightsProfile weights_profile);

// `peer_model` is rank 1's own model view at tp == 2 and nullptr at tp == 1; `execution` supplies
// the device contexts (one or two). The two must agree: a non-null peer view with a tp1 execution
// context, or the reverse, is rejected.
template <class Variant>
[[nodiscard]] std::unique_ptr<Program<Variant>>
create_program(const typename Variant::ModelView& model,
               const typename Variant::ModelView* peer_model,
               typename Variant::WeightsProfile weights_profile, SequencePlan<Variant>&& plan,
               ExecutionContext& execution);

} // namespace ninfer::targets::qwen3_6
