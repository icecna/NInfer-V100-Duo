// Host-only contract test for the context-cache cost model's fixed-point pricing.

#include "runtime/engine/context_cache/context_cost.h"

#include <cstdint>
#include <iostream>

namespace {

using ninfer::runtime::ContextPrefillCost;
using ninfer::runtime::ContextTransferCost;
using ninfer::runtime::kContextCostQ32One;
using ninfer::runtime::scale_q32;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

int test_scale() {
    int failures = 0;
    failures += check(scale_q32(kContextCostQ32One, 10) == 10, "one ns/unit scales linearly");
    failures += check(scale_q32(kContextCostQ32One / 2, 8) == 4, "half ns/unit scales");
    failures += check(scale_q32(kContextCostQ32One, 0) == 0, "zero count prices zero");
    return failures;
}

int test_transfer_price() {
    int failures = 0;
    ContextTransferCost launch;
    launch.batch_ns        = 100;
    launch.operation_ns    = 10;
    launch.ns_per_byte_q32 = 0;
    failures += check(launch.price_ns(1000, 3) == 130, "launch term dominates");

    ContextTransferCost bytes;
    bytes.ns_per_byte_q32 = kContextCostQ32One; // 1 ns/byte
    failures += check(bytes.price_ns(500, 0) == 500, "byte term prices the payload");
    failures += check(bytes.price_ns(500, 100) == 500, "the larger term wins");
    return failures;
}

int test_prefill_price() {
    int failures = 0;
    ContextPrefillCost prefill;
    prefill.chunk_ns              = 1000;
    prefill.token_ns_q32          = kContextCostQ32One;       // 1 ns/token
    prefill.attention_pair_ns_q32 = kContextCostQ32One / 8;   // 0.125 ns/pair
    prefill.vision_item_ns        = 50;
    prefill.vision_patch_ns_q32   = kContextCostQ32One / 64;  // 0.015625 ns/patch
    failures += check(prefill.price_ns(100, 160, 2, 256) == 1000 + 100 + 20 + 100 + 4,
                      "prefill price sums every term");
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += test_scale();
    failures += test_transfer_price();
    failures += test_prefill_price();
    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
