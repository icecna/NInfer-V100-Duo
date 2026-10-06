#pragma once

// Protocol-independent prompt-cache policy helpers shared by the OpenAI Chat Completions and
// OpenAI Responses adapters. They validate the cache control fields an OpenAI-shaped body may
// carry and turn the request's declared boundaries into normalized CacheBoundary values.
//
// The Anthropic Messages adapter keeps its own `cache_control` vocabulary and policy
// (anthropic_schema.cpp) because its boundary syntax and error codes differ.

#include "serve/request.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string_view>

namespace ninfer::serve {

// How the protocol asked the server to pick the reusable boundary when the caller did not mark one.
enum class OpenAIPromptCacheAutomatic : std::uint8_t {
    Default,   // no prompt_cache_options: apply the default automatic target
    Requested, // prompt_cache_options with mode implicit: apply the requested target
    Disabled,  // prompt_cache_options with mode explicit: explicit boundaries only
};

struct OpenAIPromptCachePolicy {
    OpenAIPromptCacheAutomatic automatic = OpenAIPromptCacheAutomatic::Default;
};

// True when `value` carries an explicit `prompt_cache_breakpoint`. Throws ApiException with code
// `invalid_cache_breakpoint` when the field is present but not `{mode:'explicit'}`.
[[nodiscard]] bool parse_openai_prompt_cache_breakpoint(const nlohmann::json& value,
                                                        std::string_view param);

// Validate the OpenAI cache control fields (prompt_cache_key, safety_identifier, user,
// prompt_cache_retention, prompt_cache_options) and report the automatic policy. Throws
// ApiException on malformed values.
[[nodiscard]] OpenAIPromptCachePolicy parse_openai_prompt_cache_policy(const nlohmann::json& body);

// Normalize declared boundaries, synthesize the automatic target, enforce the explicit-marker
// budget, and mark the request as owning its write policy.
void apply_openai_prompt_cache_policy(GenerationRequest& request, OpenAIPromptCachePolicy policy);

} // namespace ninfer::serve
