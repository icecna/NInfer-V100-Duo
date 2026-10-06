// Host-only contract test for the prefix shortlist identity: the rolling digest must summarize
// exactly the fields the authoritative prefix comparison uses, so a digest mismatch can only
// reject a candidate and a digest match still requires the exact comparison. No device is needed.

#include "targets/qwen3_6/impl/runtime/prefix_identity.h"

#include <ninfer/targets/qwen3_6/prepared_prompt.h>

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

using ninfer::targets::qwen3_6::PreparedPromptData;
using ninfer::targets::qwen3_6::PromptModality;
using ninfer::targets::qwen3_6::TokenSpan;
using ninfer::targets::qwen3_6::VisionGrid;
using ninfer::targets::qwen3_6::VisionItem;
using ninfer::targets::qwen3_6::detail::PrefixDigest;
using ninfer::targets::qwen3_6::detail::ResidentPrefixIdentity;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

PreparedPromptData make_prompt(const std::vector<int>& ids) {
    PreparedPromptData prompt;
    prompt.token_ids.assign(ids.begin(), ids.end());
    prompt.token_types.assign(ids.size(), 0);
    prompt.positions.resize(ids.size() * 3);
    for (std::size_t index = 0; index < ids.size(); ++index) {
        prompt.positions[index]                  = static_cast<std::int32_t>(index);
        prompt.positions[ids.size() + index]     = static_cast<std::int32_t>(index);
        prompt.positions[2 * ids.size() + index] = static_cast<std::int32_t>(index);
    }
    return prompt;
}

void add_image(PreparedPromptData& prompt, std::size_t begin, std::size_t count,
               std::uint8_t content_byte) {
    VisionItem item;
    item.modality = PromptModality::Image;
    item.grid     = VisionGrid{.temporal = 1, .height = 4, .width = 4};
    item.patch_begin = 0;
    item.patch_count = 16;
    item.content_digest.fill(content_byte);
    item.timestamps = {0.0};
    item.token_spans.push_back(TokenSpan{.begin = begin, .count = count});
    prompt.vision_items.push_back(std::move(item));
}

int test_prefix_consistency() {
    int failures = 0;
    const PreparedPromptData prompt = make_prompt({10, 11, 12, 13, 14});

    std::vector<PrefixDigest> digests;
    ninfer::targets::qwen3_6::detail::assign_prefix_digests(prompt, digests);
    failures += check(digests.size() == prompt.token_ids.size() + 1,
                      "digest table is sized tokens+1");

    // The rolling table agrees with a truncated prompt's final frontier.
    for (std::size_t frontier = 1; frontier <= prompt.token_ids.size(); ++frontier) {
        PreparedPromptData prefix = make_prompt(
            std::vector<int>(prompt.token_ids.begin(),
                             prompt.token_ids.begin() + static_cast<std::ptrdiff_t>(frontier)));
        std::vector<PrefixDigest> prefix_digests;
        ninfer::targets::qwen3_6::detail::assign_prefix_digests(prefix, prefix_digests);
        failures += check(digests[frontier] == prefix_digests[frontier],
                          "rolling digest disagrees with the truncated prompt");
    }

    // ResidentPrefixIdentity stores the same table.
    ResidentPrefixIdentity identity;
    identity.assign(prompt);
    failures += check(identity.size() == prompt.token_ids.size(), "identity token count");
    for (std::size_t frontier = 0; frontier <= prompt.token_ids.size(); ++frontier) {
        failures += check(identity.digest_at(frontier) == digests[frontier],
                          "resident digest disagrees with the prompt digest");
    }

    // A differing token changes exactly the frontiers after it.
    PreparedPromptData changed = prompt;
    changed.token_ids[2]       = 99;
    std::vector<PrefixDigest> changed_digests;
    ninfer::targets::qwen3_6::detail::assign_prefix_digests(changed, changed_digests);
    for (std::size_t frontier = 0; frontier <= 2; ++frontier) {
        failures += check(digests[frontier] == changed_digests[frontier],
                          "digest changed before the differing token");
    }
    for (std::size_t frontier = 3; frontier <= prompt.token_ids.size(); ++frontier) {
        failures += check(digests[frontier] != changed_digests[frontier],
                          "digest did not change after the differing token");
    }

    // Token type and each position axis participate in the digest.
    PreparedPromptData retyped = prompt;
    retyped.token_types[1]     = 1;
    std::vector<PrefixDigest> retyped_digests;
    ninfer::targets::qwen3_6::detail::assign_prefix_digests(retyped, retyped_digests);
    failures += check(digests[1] == retyped_digests[1] &&
                          digests[2] != retyped_digests[2],
                      "token type is not part of the digest");

    PreparedPromptData moved = prompt;
    moved.positions[prompt.token_ids.size() + 3] += 1; // height axis of token 3
    std::vector<PrefixDigest> moved_digests;
    ninfer::targets::qwen3_6::detail::assign_prefix_digests(moved, moved_digests);
    failures += check(digests[3] == moved_digests[3] && digests[4] != moved_digests[4],
                      "position axis is not part of the digest");
    return failures;
}

int test_media_digest() {
    int failures = 0;
    PreparedPromptData prompt = make_prompt({10, 11, 12, 13, 14, 15, 16});
    add_image(prompt, 2, 2, 0x11);

    PreparedPromptData other = prompt;
    other.vision_items[0].content_digest.fill(0x22);

    std::vector<PrefixDigest> digests;
    std::vector<PrefixDigest> other_digests;
    ninfer::targets::qwen3_6::detail::assign_prefix_digests(prompt, digests);
    ninfer::targets::qwen3_6::detail::assign_prefix_digests(other, other_digests);
    for (std::size_t frontier = 0; frontier < 4; ++frontier) {
        failures += check(digests[frontier] == other_digests[frontier],
                          "media digest changed a frontier before the item completes");
    }
    for (std::size_t frontier = 4; frontier <= prompt.token_ids.size(); ++frontier) {
        failures += check(digests[frontier] != other_digests[frontier],
                          "media content digest is not part of the identity");
    }
    return failures;
}

int test_identity_tag_and_key() {
    int failures = 0;
    const std::uint32_t base = ninfer::targets::qwen3_6::detail::make_prefix_identity_tag(
        /*speculative_backend=*/1, /*proposal_head=*/2, /*kv_dtype=*/3, /*kv_quant_group=*/64,
        /*tp=*/2, /*vision=*/true);
    failures += check(base == ninfer::targets::qwen3_6::detail::make_prefix_identity_tag(1, 2, 3,
                                                                                         64, 2,
                                                                                         true),
                      "identity tag is not deterministic");
    failures += check(base != ninfer::targets::qwen3_6::detail::make_prefix_identity_tag(
                                 0, 2, 3, 64, 2, true),
                      "speculative backend is not part of the identity tag");
    failures += check(base != ninfer::targets::qwen3_6::detail::make_prefix_identity_tag(
                                 1, 2, 3, 64, 1, true),
                      "TP width is not part of the identity tag");
    failures += check(base != ninfer::targets::qwen3_6::detail::make_prefix_identity_tag(
                                 1, 2, 3, 64, 2, false),
                      "vision mode is not part of the identity tag");

    const ninfer::targets::qwen3_6::detail::PrefixShortlistKey key{
        .digest = PrefixDigest{1, 2}, .frontier = 7, .identity_tag = base};
    ninfer::targets::qwen3_6::detail::PrefixShortlistKey same = key;
    ninfer::targets::qwen3_6::detail::PrefixShortlistKey other = key;
    other.identity_tag = base + 1;
    failures += check(key == same, "equal shortlist keys compare unequal");
    failures += check(!(key == other), "differing identity tags compare equal");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_prefix_consistency();
    failures += test_media_digest();
    failures += test_identity_tag_and_key();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
