#include "targets/qwen3_6/impl/frontend/chat_template.h"

#include "targets/qwen3_6/impl/frontend/digest.h"

#include <nlohmann/json.hpp>

#include <array>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6::frontend_internal {
namespace {

using OrderedJson = nlohmann::ordered_json;

constexpr Sha256Digest kThinkingToggleTemplateDigest{
    0xe8, 0x4f, 0x32, 0xa2, 0x3f, 0xdd, 0xa2, 0x76, 0x89, 0xf8, 0x68, 0xaa, 0x4a, 0x1a, 0x56, 0x21,
    0xf4, 0x11, 0x33, 0xe5, 0x1a, 0x48, 0xd7, 0xf3, 0xef, 0xcb, 0xea, 0x28, 0x39, 0x57, 0x42, 0x59,
};

constexpr Sha256Digest kReasoningEffortTemplateDigest{
    0xc3, 0xcf, 0x9e, 0x34, 0xab, 0xf4, 0xf9, 0xe3, 0x6c, 0x2d, 0x72, 0x16, 0x5a, 0xa9, 0xc1, 0x32,
    0xd3, 0xe2, 0xa7, 0x25, 0xb6, 0xc2, 0x58, 0x6a, 0xaa, 0x3a, 0x8a, 0xf9, 0xd7, 0xa8, 0x10, 0x41,
};

constexpr std::string_view kLowReasoningInstructions =
    "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to "
    "the conclusion without unnecessary elaboration.";

constexpr std::string_view kXHighReasoningInstructions =
    "Reasoning effort is set to xhigh. Please think carefully through the task, validate key "
    "assumptions, consider plausible alternatives, and prioritize correctness, consistency, and "
    "clarity in the final answer.";

bool is_instruction_role(ChatRole role) noexcept {
    return role == ChatRole::System || role == ChatRole::Developer;
}

void validate_instruction_message(const ChatMessage& message) {
    if (message.has_media()) {
        throw std::invalid_argument(
            "system and developer messages cannot contain images or videos");
    }
    if (!message.reasoning_content.empty() || !message.tool_calls.empty() ||
        !message.tool_call_id.empty()) {
        throw std::invalid_argument("system and developer messages may contain only text content");
    }
}

std::string trim_ascii_whitespace(const std::string& text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }

    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return text.substr(begin, end - begin);
}

struct TrimBounds {
    std::size_t begin = 0;
    std::size_t end   = 0;
};

// Byte range of the message content that survives trim_ascii_whitespace. A cache boundary outside
// [begin, end] falls inside trimmed whitespace and therefore has no output position.
TrimBounds ascii_trim_bounds(const std::string& text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])) != 0) { --end; }
    return TrimBounds{begin, end};
}

bool starts_with(const std::string& text, std::string_view prefix) {
    return text.size() >= prefix.size() && text.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& text, std::string_view suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

long last_real_user_query(const std::vector<ChatMessage>& messages) {
    for (long i = static_cast<long>(messages.size()) - 1; i >= 0; --i) {
        const ChatMessage& message = messages[static_cast<std::size_t>(i)];
        if (message.role != ChatRole::User) { continue; }
        const std::string content = trim_ascii_whitespace(message.rendered_content());
        if (!(starts_with(content, "<tool_response>") && ends_with(content, "</tool_response>"))) {
            return i;
        }
    }
    throw std::invalid_argument("no user query found in chat messages");
}

std::string lstrip_newlines(std::string text) {
    std::size_t begin = 0;
    while (begin < text.size() && text[begin] == '\n') { ++begin; }
    return text.substr(begin);
}

std::string rstrip_newlines(std::string text) {
    std::size_t end = text.size();
    while (end > 0 && text[end - 1] == '\n') { --end; }
    return text.substr(0, end);
}

// Split an assistant turn into (reasoning, content) exactly as the Qwen3.6 jinja
// does when reasoning_content is not provided: reasoning is the text between the
// last <think> and the first </think>; content is everything after the last
// </think>. When there is no </think> the whole thing is content and reasoning is
// empty.
struct ThinkParts {
    std::string reasoning;
    std::string content;
};

ThinkParts derive_think_parts(const std::string& content) {
    ThinkParts parts;
    const std::size_t first_close = content.find("</think>");
    if (first_close == std::string::npos) {
        parts.content = content;
        return parts;
    }
    // reasoning = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')
    std::string before          = rstrip_newlines(content.substr(0, first_close));
    const std::size_t last_open = before.rfind("<think>");
    std::string reasoning       = (last_open == std::string::npos)
                                      ? before
                                      : before.substr(last_open + std::string("<think>").size());
    parts.reasoning             = lstrip_newlines(std::move(reasoning));
    // content = content.split('</think>')[-1].lstrip('\n')
    const std::size_t last_close = content.rfind("</think>");
    parts.content = lstrip_newlines(content.substr(last_close + std::string("</think>").size()));
    return parts;
}

constexpr std::string_view kToolInstructions =
    "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n"
    "<tool_call>\n"
    "<function=example_function_name>\n"
    "<parameter=example_parameter_1>\n"
    "value_1\n"
    "</parameter>\n"
    "<parameter=example_parameter_2>\n"
    "This is the value for the second parameter\n"
    "that can span\n"
    "multiple lines\n"
    "</parameter>\n"
    "</function>\n"
    "</tool_call>\n\n"
    "<IMPORTANT>\n"
    "Reminder:\n"
    "- Function calls MUST follow the specified format: an inner <function=...></function> block "
    "must be nested within <tool_call></tool_call> XML tags\n"
    "- Required parameters MUST be specified\n"
    "- You may provide optional reasoning for your function call in natural language BEFORE the "
    "function call, but NOT after\n"
    "- If there is no function call available, answer the question like normal with your current "
    "knowledge and do not tell the user about function calls\n"
    "</IMPORTANT>";

std::string tojson_text(const OrderedJson& value) {
    if (value.is_array()) {
        std::string rendered = "[";
        for (std::size_t index = 0; index < value.size(); ++index) {
            if (index != 0) { rendered += ", "; }
            rendered += tojson_text(value[index]);
        }
        rendered += "]";
        return rendered;
    }
    if (value.is_object()) {
        std::string rendered = "{";
        std::size_t index    = 0;
        for (auto it = value.begin(); it != value.end(); ++it, ++index) {
            if (index != 0) { rendered += ", "; }
            rendered += OrderedJson(it.key()).dump();
            rendered += ": ";
            rendered += tojson_text(it.value());
        }
        rendered += "}";
        return rendered;
    }
    return value.dump();
}

std::string parameter_text(const OrderedJson& value) {
    if (value.is_string()) { return value.get<std::string>(); }
    return tojson_text(value);
}

std::string render_tool_call(const ToolCall& call, bool allow_empty_arguments) {
    if (allow_empty_arguments && call.arguments_json.empty()) {
        return "<tool_call>\n<function=" + call.name + ">\n</function>\n</tool_call>";
    }
    OrderedJson args = OrderedJson::parse(call.arguments_json);
    if (!args.is_object()) {
        throw std::invalid_argument("tool call arguments must be a JSON object");
    }

    std::string rendered;
    rendered += "<tool_call>\n<function=";
    rendered += call.name;
    rendered += ">\n";
    for (auto it = args.begin(); it != args.end(); ++it) {
        rendered += "<parameter=";
        rendered += it.key();
        rendered += ">\n";
        rendered += parameter_text(it.value());
        rendered += "\n</parameter>\n";
    }
    rendered += "</function>\n</tool_call>";
    return rendered;
}

struct ToolsSystemBlock {
    std::string text;
    std::vector<std::size_t> tool_ends; // end offset of each tool definition within text
    std::size_t instruction_start = 0;
    bool has_instruction          = false;
};

ToolsSystemBlock render_tools_system_block(const std::vector<std::string>& tool_jsons,
                                           const std::string& leading_instruction,
                                           std::string_view reasoning_instructions) {
    ToolsSystemBlock block;
    std::string& rendered = block.text;
    rendered += "<|im_start|>system\n";
    if (!reasoning_instructions.empty()) {
        rendered += reasoning_instructions;
        rendered += "\n\n";
    }
    rendered += "# Tools\n\nYou have access to the following functions:\n\n<tools>";
    block.tool_ends.reserve(tool_jsons.size());
    for (const std::string& tool : tool_jsons) {
        rendered += "\n";
        rendered += tojson_text(OrderedJson::parse(tool));
        block.tool_ends.push_back(rendered.size());
    }
    rendered += "\n</tools>";
    rendered += std::string(kToolInstructions);
    if (!leading_instruction.empty()) {
        rendered += "\n\n";
        block.instruction_start = rendered.size();
        block.has_instruction   = true;
        rendered += leading_instruction;
    }
    rendered += "<|im_end|>\n";
    return block;
}

std::string_view resolve_reasoning_instructions(ChatTemplateSemantics semantics,
                                                const ChatRenderOptions& options) {
    if (semantics == ChatTemplateSemantics::ThinkingToggle) {
        if (options.reasoning_effort) {
            throw std::invalid_argument("loaded chat template does not support reasoning effort");
        }
        return {};
    }
    if (!options.enable_thinking) {
        if (options.reasoning_effort) {
            throw std::invalid_argument(
                "reasoning effort cannot be combined with disabled thinking");
        }
        return {};
    }

    switch (options.reasoning_effort.value_or(ReasoningEffort::XHigh)) {
    case ReasoningEffort::Low:
        return kLowReasoningInstructions;
    case ReasoningEffort::Medium:
        return {};
    case ReasoningEffort::XHigh:
        return kXHighReasoningInstructions;
    }
    throw std::invalid_argument("invalid reasoning effort");
}

} // namespace

bool ChatMessage::has_media() const noexcept {
    for (const ChatPart& part : parts) {
        if (part.kind != ChatPartKind::Text) { return true; }
    }
    return false;
}

ChatMessage::RenderedContentParts ChatMessage::rendered_content_parts(bool add_vision_id,
                                                                      int* image_count,
                                                                      int* video_count) const {
    int local_images = 0;
    int local_videos = 0;
    int& images      = image_count == nullptr ? local_images : *image_count;
    int& videos      = video_count == nullptr ? local_videos : *video_count;
    RenderedContentParts out;
    out.part_ends.reserve(parts.size());
    for (const ChatPart& part : parts) {
        switch (part.kind) {
        case ChatPartKind::Text:
            out.text += part.text;
            break;
        case ChatPartKind::Image:
            ++images;
            if (add_vision_id) { out.text += "Picture " + std::to_string(images) + ": "; }
            out.text += "<|vision_start|><|image_pad|><|vision_end|>";
            break;
        case ChatPartKind::Video:
            ++videos;
            if (add_vision_id) { out.text += "Video " + std::to_string(videos) + ": "; }
            out.text += "<|vision_start|><|video_pad|><|vision_end|>";
            break;
        }
        out.part_ends.push_back(out.text.size());
    }
    return out;
}

std::string ChatMessage::rendered_content(bool add_vision_id, int* image_count,
                                          int* video_count) const {
    return rendered_content_parts(add_vision_id, image_count, video_count).text;
}

CompiledChatTemplate CompiledChatTemplate::resolve(std::string_view source) {
    const Sha256Digest digest = sha256(source);
    if (digest == kThinkingToggleTemplateDigest) {
        return CompiledChatTemplate(ChatTemplateSemantics::ThinkingToggle);
    }
    if (digest == kReasoningEffortTemplateDigest) {
        return CompiledChatTemplate(ChatTemplateSemantics::ReasoningEffort);
    }
    throw std::invalid_argument("unsupported frontend/chat_template.jinja (sha256 " +
                                sha256_hex(digest) + ")");
}

PromptCapabilities CompiledChatTemplate::capabilities() const noexcept {
    PromptCapabilities result;
    result.enable_thinking = true;
    if (semantics_ == ChatTemplateSemantics::ReasoningEffort) {
        result.reasoning_effort.low            = true;
        result.reasoning_effort.medium         = true;
        result.reasoning_effort.xhigh          = true;
        result.reasoning_effort.default_effort = ReasoningEffort::XHigh;
    }
    return result;
}

RenderedChat CompiledChatTemplate::render(const std::vector<ChatMessage>& messages,
                                          ChatRenderOptions options) const {
    if (messages.empty()) { throw std::invalid_argument("chat messages must not be empty"); }

    const bool effort_template = semantics_ == ChatTemplateSemantics::ReasoningEffort;
    const std::string_view reasoning_instructions =
        resolve_reasoning_instructions(semantics_, options);

    constexpr std::size_t kNoOffset = std::numeric_limits<std::size_t>::max();

    std::size_t message_begin = 0;
    std::string leading_instruction;
    TrimBounds leading_trim;
    ChatMessage::RenderedContentParts leading_parts;
    std::size_t instruction_start = kNoOffset;
    if (is_instruction_role(messages[0].role)) {
        validate_instruction_message(messages[0]);
        leading_parts = messages[0].rendered_content_parts(options.add_vision_id);
        leading_trim  = ascii_trim_bounds(leading_parts.text);
        leading_instruction =
            leading_parts.text.substr(leading_trim.begin, leading_trim.end - leading_trim.begin);
        message_begin = 1;
    }

    std::string rendered;
    std::vector<std::size_t> message_end_offset(messages.size(), kNoOffset);
    std::vector<std::vector<std::size_t>> part_end_offset(messages.size());
    std::vector<std::size_t> tool_ends;

    const auto map_part_offsets = [&](std::size_t message_index,
                                      const ChatMessage::RenderedContentParts& content_parts,
                                      const TrimBounds& trim, std::size_t content_start,
                                      bool enabled) {
        part_end_offset[message_index].assign(content_parts.part_ends.size(), kNoOffset);
        if (!enabled || content_start == kNoOffset) { return; }
        for (std::size_t index = 0; index < content_parts.part_ends.size(); ++index) {
            const std::size_t cumulative = content_parts.part_ends[index];
            if (cumulative < trim.begin || cumulative > trim.end) { continue; }
            part_end_offset[message_index][index] = content_start + (cumulative - trim.begin);
        }
    };

    const bool has_tools = !options.tool_jsons.empty();
    if (has_tools) {
        const ToolsSystemBlock block =
            render_tools_system_block(options.tool_jsons, leading_instruction,
                                      reasoning_instructions);
        rendered = block.text;
        tool_ends = block.tool_ends;
        if (block.has_instruction) {
            instruction_start = block.instruction_start;
            map_part_offsets(0, leading_parts, leading_trim, instruction_start, true);
        }
        if (message_begin == 1) { message_end_offset[0] = rendered.size(); }
    } else if (message_begin == 1) {
        if (!effort_template || !leading_instruction.empty() || !reasoning_instructions.empty()) {
            rendered += "<|im_start|>system\n";
            if (!reasoning_instructions.empty()) {
                rendered += reasoning_instructions;
                if (!leading_instruction.empty()) { rendered += "\n\n"; }
            }
            instruction_start = rendered.size();
            rendered += leading_instruction;
            rendered += "<|im_end|>\n";
            map_part_offsets(0, leading_parts, leading_trim, instruction_start, true);
        }
        message_end_offset[0] = rendered.size();
    } else if (!reasoning_instructions.empty()) {
        rendered += "<|im_start|>system\n";
        rendered += reasoning_instructions;
        rendered += "<|im_end|>\n";
    }

    const long last_query_index  = last_real_user_query(messages);
    const bool preserve_thinking = options.preserve_thinking.value_or(effort_template);
    std::optional<RewriteCheckpointByteSpec> rewrite_checkpoint;

    int image_count = 0;
    int video_count = 0;
    for (std::size_t i = 0; i < messages.size(); ++i) {
        const ChatMessage& message = messages[i];
        if (i < message_begin) { continue; }
        if (is_instruction_role(message.role)) { validate_instruction_message(message); }
        const ChatMessage::RenderedContentParts content_parts =
            message.rendered_content_parts(options.add_vision_id, &image_count, &video_count);
        const TrimBounds trim = ascii_trim_bounds(content_parts.text);
        const std::string content =
            content_parts.text.substr(trim.begin, trim.end - trim.begin);
        if (is_instruction_role(message.role)) {
            rendered += "<|im_start|>system\n";
            const std::size_t content_start = rendered.size();
            rendered += content;
            map_part_offsets(i, content_parts, trim, content_start, true);
            rendered += "<|im_end|>\n";
            message_end_offset[i] = rendered.size();
            continue;
        }
        if (message.role == ChatRole::User) {
            rendered += "<|im_start|>user\n";
            const std::size_t content_start = rendered.size();
            rendered += content;
            map_part_offsets(i, content_parts, trim, content_start, true);
            rendered += "<|im_end|>\n";
            message_end_offset[i] = rendered.size();
            continue;
        }
        if (message.role == ChatRole::Tool) {
            const bool opens_group = i > 0 && messages[i - 1].role != ChatRole::Tool;
            const bool closes_group =
                i + 1 == messages.size() || messages[i + 1].role != ChatRole::Tool;
            if (opens_group) { rendered += "<|im_start|>user"; }
            rendered += "\n<tool_response>\n";
            const std::size_t content_start = rendered.size();
            rendered += content;
            map_part_offsets(i, content_parts, trim, content_start, true);
            rendered += "\n</tool_response>";
            if (closes_group) { rendered += "<|im_end|>\n"; }
            message_end_offset[i] = rendered.size();
            continue;
        }

        if (message.role != ChatRole::Assistant) {
            throw std::invalid_argument("unsupported chat role value");
        }

        // assistant
        std::string reasoning;
        std::string body = content;
        if (!message.reasoning_content.empty()) {
            reasoning = message.reasoning_content;
        } else if (!effort_template) {
            ThinkParts parts = derive_think_parts(content);
            reasoning        = std::move(parts.reasoning);
            body             = std::move(parts.content);
        }
        reasoning = trim_ascii_whitespace(reasoning);

        const bool keep_thinking = preserve_thinking || (static_cast<long>(i) > last_query_index);
        rendered += "<|im_start|>assistant\n";
        if (!preserve_thinking && !rewrite_checkpoint && static_cast<long>(i) > last_query_index) {
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::TurnClosure, .offset = rendered.size()};
        }
        if (keep_thinking) {
            rendered += "<think>\n";
            rendered += reasoning;
            rendered += "\n</think>\n\n";
        }
        const std::size_t content_start = rendered.size();
        rendered += body;
        // Only an assistant turn whose body is emitted verbatim at a known offset has a resolvable
        // message-part boundary; a derived think split or an inserted prologue is dropped.
        map_part_offsets(i, content_parts, trim, content_start,
                         body == content && !keep_thinking);
        if (!message.tool_calls.empty()) {
            const bool body_has_text = !trim_ascii_whitespace(body).empty();
            for (std::size_t call_index = 0; call_index < message.tool_calls.size(); ++call_index) {
                if (call_index == 0) {
                    if (body_has_text) { rendered += "\n\n"; }
                } else {
                    rendered += "\n";
                }
                rendered += render_tool_call(message.tool_calls[call_index], effort_template);
            }
        }
        rendered += "<|im_end|>\n";
        message_end_offset[i] = rendered.size();
    }

    if (options.add_generation_prompt) {
        rendered += "<|im_start|>assistant\n";
        // The generation prompt is the frontier every follow-up replay reaches, so it is the
        // right anchor even when an earlier assistant turn (a tool call) already claimed one: that
        // earlier boundary sits before the tool results and is unreachable for the next request,
        // which is what left the host cache holding huge terminal residencies.
        if (!preserve_thinking) {
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::TurnClosure, .offset = rendered.size()};
        }
        if (options.enable_thinking) {
            rendered += "<think>\n";
        } else {
            rendered += "<think>\n\n</think>\n\n";
        }
        if (preserve_thinking) {
            // Response replay retains the deterministic generation prologue. This is the prompt
            // frontier for both thinking modes, so capturing it does not split off a tiny final
            // prefill unit. The complete rendered prefix is tokenized independently below.
            rewrite_checkpoint = RewriteCheckpointByteSpec{
                .kind = RewriteCheckpointKind::ResponseReplay, .offset = rendered.size()};
        }
    }
    RenderedChat result;
    result.text               = std::move(rendered);
    result.rewrite_checkpoint = rewrite_checkpoint;

    std::vector<ninfer::PromptCacheMarker> markers = options.cache_markers;
    if (options.add_engine_automatic_shared_prefixes) {
        // Structural candidates: the end of the tool definitions and the end of the leading
        // instruction, both of which a caller reproducing the conversation can resend verbatim.
        if (!tool_ends.empty()) {
            markers.push_back(ninfer::PromptCacheMarker{
                .kind             = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                .evidence         = ninfer::SharedCandidateEvidence::EngineStructural,
                .location         = ninfer::PromptCacheMarkerLocation::ToolBoundary,
                .after_tool_count = static_cast<std::uint32_t>(tool_ends.size()),
            });
        }
        if (message_begin == 1 && instruction_start != kNoOffset) {
            markers.push_back(ninfer::PromptCacheMarker{
                .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
                .evidence = ninfer::SharedCandidateEvidence::EngineStructural,
                .location = ninfer::PromptCacheMarkerLocation::LeadingInstructionBoundary,
                .leading_instruction_bytes = static_cast<std::uint32_t>(leading_trim.end),
            });
        }
    }

    result.cache_markers = markers;
    result.cache_boundary_offsets.reserve(markers.size());
    for (const ninfer::PromptCacheMarker& marker : markers) {
        std::optional<std::size_t> offset;
        switch (marker.location) {
        case ninfer::PromptCacheMarkerLocation::MessageBoundary:
            if (marker.after_message_count >= 1 && marker.after_message_count <= messages.size()) {
                const std::size_t value = message_end_offset[marker.after_message_count - 1];
                if (value != kNoOffset) { offset = value; }
            }
            break;
        case ninfer::PromptCacheMarkerLocation::MessagePartBoundary:
            if (marker.after_message_count >= 1 && marker.after_message_count <= messages.size() &&
                marker.after_message_part_count >= 1) {
                const std::vector<std::size_t>& ends =
                    part_end_offset[marker.after_message_count - 1];
                if (marker.after_message_part_count <= ends.size()) {
                    const std::size_t value = ends[marker.after_message_part_count - 1];
                    if (value != kNoOffset) { offset = value; }
                }
            }
            break;
        case ninfer::PromptCacheMarkerLocation::ToolBoundary:
            if (marker.after_tool_count >= 1 && marker.after_tool_count <= tool_ends.size()) {
                offset = tool_ends[marker.after_tool_count - 1];
            }
            break;
        case ninfer::PromptCacheMarkerLocation::LeadingInstructionBoundary:
            if (instruction_start != kNoOffset &&
                marker.leading_instruction_bytes >= leading_trim.begin &&
                marker.leading_instruction_bytes <= leading_trim.end) {
                offset =
                    instruction_start + (marker.leading_instruction_bytes - leading_trim.begin);
            }
            break;
        }
        result.cache_boundary_offsets.push_back(offset);
    }
    return result;
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
