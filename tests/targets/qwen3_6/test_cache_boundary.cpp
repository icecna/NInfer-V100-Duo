// Host-only contract test for renderer cache-boundary resolution: each PromptCacheMarker location
// resolves to the byte offset its structural position implies in the rendered chat, and a boundary
// without a unique output position is dropped rather than approximated. No tokenizer or device is
// needed, so this runs in the default test environment.

#include "targets/qwen3_6/impl/frontend/chat_template.h"

#include <ninfer/types.h>

#include <cstddef>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fi = ninfer::targets::qwen3_6::frontend_internal;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

std::string read_template_fixture(const char* path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) { throw std::runtime_error(std::string("failed to open fixture: ") + path); }
    std::string source{std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
    if (!source.empty() && source.back() == '\n') { source.pop_back(); }
    return source;
}

const fi::CompiledChatTemplate& thinking_toggle_template() {
    static const fi::CompiledChatTemplate value = fi::CompiledChatTemplate::resolve(
        read_template_fixture(NINFER_SOURCE_DIR
                              "/tests/fixtures/frontend/thinking_toggle_chat_template.jinja"));
    return value;
}

fi::ChatMessage chat_message(ninfer::ChatRole role, std::string content) {
    fi::ChatMessage message;
    message.role = role;
    message.parts.push_back(fi::ChatPart::text_part(std::move(content)));
    return message;
}

fi::RenderedChat render_chat(std::vector<fi::ChatMessage> messages, fi::ChatRenderOptions options) {
    return thinking_toggle_template().render(messages, std::move(options));
}

int test_message_and_part_boundaries() {
    int failures             = 0;
    const std::string im_end = "<|im_end|>\n";

    {
        fi::ChatRenderOptions options;
        options.cache_markers.push_back(ninfer::PromptCacheMarker{
            .after_message_count = 2,
            .location            = ninfer::PromptCacheMarkerLocation::MessageBoundary,
        });
        const fi::RenderedChat rendered =
            render_chat({chat_message(ninfer::ChatRole::System, "sys"),
                         chat_message(ninfer::ChatRole::User, "hi")},
                        options);
        failures += check(rendered.cache_boundary_offsets.size() == 1 &&
                              rendered.cache_boundary_offsets[0].has_value(),
                          "message boundary was not resolved");
        if (rendered.cache_boundary_offsets.size() == 1 && rendered.cache_boundary_offsets[0]) {
            const std::string prefix =
                rendered.text.substr(0, *rendered.cache_boundary_offsets[0]);
            failures += check(prefix.ends_with(im_end) && prefix.find("hi") != std::string::npos,
                              "message boundary is not after the first user turn");
        }
    }

    {
        fi::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        user.parts.push_back(fi::ChatPart::text_part("alpha"));
        user.parts.push_back(fi::ChatPart::text_part("\n"));
        user.parts.push_back(fi::ChatPart::text_part("beta"));
        fi::ChatRenderOptions options;
        options.cache_markers.push_back(ninfer::PromptCacheMarker{
            .after_message_count      = 1,
            .location                 = ninfer::PromptCacheMarkerLocation::MessagePartBoundary,
            .after_message_part_count = 1,
        });
        const fi::RenderedChat rendered = render_chat({user}, options);
        failures += check(rendered.cache_boundary_offsets.size() == 1 &&
                              rendered.cache_boundary_offsets[0].has_value() &&
                              rendered.text.substr(0, *rendered.cache_boundary_offsets[0])
                                  .ends_with("alpha"),
                          "message-part boundary is not after the first part");
    }

    {
        fi::ChatRenderOptions options;
        options.cache_markers.push_back(ninfer::PromptCacheMarker{
            .after_message_count      = 1,
            .location                 = ninfer::PromptCacheMarkerLocation::MessagePartBoundary,
            .after_message_part_count = 9,
        });
        options.cache_markers.push_back(ninfer::PromptCacheMarker{
            .location         = ninfer::PromptCacheMarkerLocation::ToolBoundary,
            .after_tool_count = 3,
        });
        const fi::RenderedChat rendered = render_chat(
            {chat_message(ninfer::ChatRole::User, "hi")}, options);
        failures += check(rendered.cache_boundary_offsets.size() == 2 &&
                              !rendered.cache_boundary_offsets[0].has_value() &&
                              !rendered.cache_boundary_offsets[1].has_value(),
                          "unresolvable boundaries were not dropped");
    }
    return failures;
}

int test_tool_and_instruction_boundaries() {
    int failures = 0;

    {
        fi::ChatRenderOptions options;
        options.tool_jsons.push_back(R"({"type":"function","function":{"name":"f"}})");
        options.cache_markers.push_back(ninfer::PromptCacheMarker{
            .kind             = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence         = ninfer::SharedCandidateEvidence::ExplicitBoundary,
            .location         = ninfer::PromptCacheMarkerLocation::ToolBoundary,
            .after_tool_count = 1,
        });
        const fi::RenderedChat rendered =
            render_chat({chat_message(ninfer::ChatRole::User, "hi")}, options);
        failures += check(rendered.cache_boundary_offsets.size() == 1 &&
                              rendered.cache_boundary_offsets[0].has_value(),
                          "tool boundary was not resolved");
        if (rendered.cache_boundary_offsets.size() == 1 && rendered.cache_boundary_offsets[0]) {
            const std::string prefix =
                rendered.text.substr(0, *rendered.cache_boundary_offsets[0]);
            failures += check(prefix.find("\"name\": \"f\"") != std::string::npos &&
                                  prefix.find("</tools>") == std::string::npos,
                              "tool boundary is not after the tool definition");
        }
    }

    {
        fi::ChatRenderOptions options;
        options.cache_markers.push_back(ninfer::PromptCacheMarker{
            .kind                      = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence                  = ninfer::SharedCandidateEvidence::ExplicitBoundary,
            .location                  = ninfer::PromptCacheMarkerLocation::LeadingInstructionBoundary,
            .leading_instruction_bytes = 3,
        });
        const fi::RenderedChat rendered =
            render_chat({chat_message(ninfer::ChatRole::System, "sys"),
                         chat_message(ninfer::ChatRole::User, "hi")},
                        options);
        failures += check(rendered.cache_boundary_offsets.size() == 1 &&
                              rendered.cache_boundary_offsets[0].has_value() &&
                              rendered.text.substr(0, *rendered.cache_boundary_offsets[0])
                                  .ends_with("sys"),
                          "leading-instruction boundary is not after the instruction");

        // A byte offset inside the trimmed trailing whitespace has no output position.
        fi::ChatRenderOptions trailing;
        trailing.cache_markers.push_back(ninfer::PromptCacheMarker{
            .kind                      = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence                  = ninfer::SharedCandidateEvidence::ExplicitBoundary,
            .location                  = ninfer::PromptCacheMarkerLocation::LeadingInstructionBoundary,
            .leading_instruction_bytes = 5,
        });
        const fi::RenderedChat trimmed =
            render_chat({chat_message(ninfer::ChatRole::System, "sys  "),
                         chat_message(ninfer::ChatRole::User, "hi")},
                        trailing);
        failures += check(trimmed.cache_boundary_offsets.size() == 1 &&
                              !trimmed.cache_boundary_offsets[0].has_value(),
                          "boundary inside trimmed whitespace was not dropped");
    }
    return failures;
}

int test_structural_candidates() {
    int failures = 0;

    {
        fi::ChatRenderOptions options;
        options.add_engine_automatic_shared_prefixes = true;
        const fi::RenderedChat rendered =
            render_chat({chat_message(ninfer::ChatRole::System, "sys"),
                         chat_message(ninfer::ChatRole::User, "hi")},
                        options);
        failures += check(rendered.cache_markers.size() == 1 &&
                              rendered.cache_markers[0].evidence ==
                                  ninfer::SharedCandidateEvidence::EngineStructural &&
                              rendered.cache_markers[0].location ==
                                  ninfer::PromptCacheMarkerLocation::LeadingInstructionBoundary &&
                              rendered.cache_boundary_offsets.size() == 1 &&
                              rendered.cache_boundary_offsets[0].has_value(),
                          "structural leading-instruction candidate was not published");
    }

    {
        fi::ChatRenderOptions options;
        options.add_engine_automatic_shared_prefixes = true;
        options.tool_jsons.push_back(R"({"type":"function","function":{"name":"f"}})");
        const fi::RenderedChat rendered =
            render_chat({chat_message(ninfer::ChatRole::User, "hi")}, options);
        failures += check(rendered.cache_markers.size() == 1 &&
                              rendered.cache_markers[0].location ==
                                  ninfer::PromptCacheMarkerLocation::ToolBoundary,
                          "structural tool candidate was not published for a tool-less system "
                          "prompt");
    }
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_message_and_part_boundaries();
    failures += test_tool_and_instruction_boundaries();
    failures += test_structural_candidates();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
