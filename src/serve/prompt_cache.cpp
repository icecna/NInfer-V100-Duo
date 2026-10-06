#include "serve/prompt_cache.h"

#include <ninfer/types.h>

#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;

[[noreturn]] void bad_request(std::string message, std::string param = {}, std::string code = {}) {
    ApiError error;
    error.status  = 400;
    error.type    = "invalid_request_error";
    error.message = std::move(message);
    error.param   = std::move(param);
    error.code    = std::move(code);
    throw ApiException(std::move(error));
}

} // namespace

bool parse_openai_prompt_cache_breakpoint(const Json& value, std::string_view param) {
    if (!value.contains("prompt_cache_breakpoint") ||
        value.at("prompt_cache_breakpoint").is_null()) {
        return false;
    }
    const Json& breakpoint = value.at("prompt_cache_breakpoint");
    if (!breakpoint.is_object() || !breakpoint.contains("mode") ||
        !breakpoint.at("mode").is_string() ||
        breakpoint.at("mode").get<std::string>() != "explicit") {
        bad_request("prompt_cache_breakpoint must be {mode:'explicit'}", std::string(param),
                    "invalid_cache_breakpoint");
    }
    return true;
}

OpenAIPromptCachePolicy parse_openai_prompt_cache_policy(const Json& body) {
    auto require_string_hint = [&](const char* field, std::optional<std::size_t> maximum = {}) {
        if (!body.contains(field) || body.at(field).is_null()) { return; }
        if (!body.at(field).is_string()) {
            bad_request(std::string(field) + " must be a string", field);
        }
        if (maximum && body.at(field).get_ref<const std::string&>().size() > *maximum) {
            bad_request(std::string(field) + " must be at most " + std::to_string(*maximum) +
                            " characters",
                        field);
        }
    };
    require_string_hint("prompt_cache_key", 64U);
    require_string_hint("safety_identifier", 64U);
    require_string_hint("user");

    if (body.contains("prompt_cache_retention") && !body.at("prompt_cache_retention").is_null()) {
        if (!body.at("prompt_cache_retention").is_string()) {
            bad_request("prompt_cache_retention must be a string", "prompt_cache_retention");
        }
        const std::string value = body.at("prompt_cache_retention").get<std::string>();
        if (value != "in_memory" && value != "24h") {
            bad_request("prompt_cache_retention must be 'in_memory' or '24h'",
                        "prompt_cache_retention");
        }
    }

    OpenAIPromptCachePolicy policy;
    if (!body.contains("prompt_cache_options") || body.at("prompt_cache_options").is_null()) {
        return policy;
    }
    const Json& options = body.at("prompt_cache_options");
    if (!options.is_object()) {
        bad_request("prompt_cache_options must be an object", "prompt_cache_options");
    }
    policy.automatic = OpenAIPromptCacheAutomatic::Requested;
    if (options.contains("mode") && !options.at("mode").is_null()) {
        if (!options.at("mode").is_string()) {
            bad_request("prompt_cache_options.mode must be a string", "prompt_cache_options");
        }
        const std::string mode = options.at("mode").get<std::string>();
        if (mode == "explicit") {
            policy.automatic = OpenAIPromptCacheAutomatic::Disabled;
        } else if (mode != "implicit") {
            bad_request("prompt_cache_options.mode must be 'implicit' or 'explicit'",
                        "prompt_cache_options");
        }
    }
    if (options.contains("ttl") && !options.at("ttl").is_null() &&
        (!options.at("ttl").is_string() || options.at("ttl").get<std::string>() != "30m")) {
        bad_request("prompt_cache_options.ttl must be '30m'", "prompt_cache_options");
    }
    return policy;
}

void apply_openai_prompt_cache_policy(GenerationRequest& request, OpenAIPromptCachePolicy policy) {
    std::vector<std::optional<CacheBoundary>*> explicit_boundaries;
    for (ToolDefinition& tool : request.tools) {
        if (tool.cache_boundary_after) { explicit_boundaries.push_back(&tool.cache_boundary_after); }
    }
    for (ChatTurn& turn : request.messages) {
        for (ContentPart& part : turn.content) {
            if (part.cache_boundary_after) {
                explicit_boundaries.push_back(&part.cache_boundary_after);
            }
        }
        if (turn.cache_boundary_after) {
            explicit_boundaries.push_back(&turn.cache_boundary_after);
        }
    }

    // The automatic target is the end of the last turn the caller is expected to resend verbatim:
    // an assistant turn that produced tool calls, else the last turn carrying content, else the
    // tool-definition block.
    std::optional<CacheBoundary>* automatic_target = nullptr;
    for (auto turn = request.messages.rbegin(); turn != request.messages.rend(); ++turn) {
        if (!turn->tool_calls.empty()) {
            automatic_target = &turn->cache_boundary_after;
            break;
        }
        if (!turn->content.empty()) {
            automatic_target = &turn->content.back().cache_boundary_after;
            break;
        }
    }
    if (automatic_target == nullptr && !request.tools.empty()) {
        automatic_target = &request.tools.back().cache_boundary_after;
    }

    const bool automatic_enabled =
        policy.automatic != OpenAIPromptCacheAutomatic::Disabled && automatic_target != nullptr &&
        (automatic_target->has_value() ||
         explicit_boundaries.size() < ninfer::kMaximumExplicitPromptCacheMarkers);
    const bool automatic_merges_explicit   = automatic_enabled && automatic_target->has_value();
    const std::size_t explicit_write_slots = !automatic_enabled || automatic_merges_explicit
                                                 ? ninfer::kMaximumExplicitPromptCacheMarkers
                                                 : ninfer::kMaximumExplicitPromptCacheMarkers - 1U;
    const std::size_t first_selected       = explicit_boundaries.size() > explicit_write_slots
                                                 ? explicit_boundaries.size() - explicit_write_slots
                                                 : 0U;
    for (std::size_t index = 0; index < explicit_boundaries.size(); ++index) {
        if (index < first_selected) {
            explicit_boundaries[index]->reset();
        } else {
            explicit_boundaries[index]->value().kind =
                ninfer::PromptCacheMarkerKind::SharedStablePrefix;
            explicit_boundaries[index]->value().evidence =
                ninfer::SharedCandidateEvidence::ExplicitBoundary;
        }
    }

    if (automatic_enabled) {
        const ninfer::SharedCandidateEvidence evidence =
            policy.automatic == OpenAIPromptCacheAutomatic::Default
                ? ninfer::SharedCandidateEvidence::DefaultAutomatic
                : ninfer::SharedCandidateEvidence::RequestedAutomatic;
        if (*automatic_target) {
            automatic_target->value().evidence |= evidence;
        } else {
            *automatic_target = CacheBoundary{.evidence = evidence};
        }
    }
    // OpenAI already defines the automatic/explicit write policy for every request. Existing exact
    // shared residents are still considered by the Engine independently of this switch.
    request.allow_engine_automatic_shared_prefixes = false;
}

} // namespace ninfer::serve
