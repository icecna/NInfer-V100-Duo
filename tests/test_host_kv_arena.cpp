#include "core/host_kv_arena.h"
#include "core/paged_kv_cache.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace ninfer;

int failures = 0;

int fail(const std::string& message) {
    std::cerr << message << '\n';
    return 1;
}

template <class Fn>
int expect_throws(Fn&& fn, const std::string& label) {
    try {
        fn();
    } catch (const std::exception&) { return 0; }
    return fail(label + " did not throw");
}

int expect(bool condition, const std::string& label) {
    return condition ? 0 : fail(label);
}

int expect_eq(std::size_t actual, std::size_t expected, const std::string& label) {
    if (actual == expected) { return 0; }
    std::cerr << label << " expected " << expected << ", got " << actual << '\n';
    return 1;
}

PagedKVPoolSpec make_pool_spec(PagedKVPlaneOrder order, std::vector<PagedKVPlaneSpec> planes) {
    PagedKVPoolSpec spec;
    spec.page_group_count = 256;
    spec.logical_page_capacity = 16384;
    spec.table_rows = 8;
    spec.plane_order = order;
    spec.planes = std::move(planes);
    return spec;
}

// A 4-plane quantized text-KV pool at one tp rank: k/v I8 groups plus fp16 scales.
PagedKVPoolSpec quantized_text_spec() {
    return make_pool_spec(PagedKVPlaneOrder::PageMajor,
                          {PagedKVPlaneSpec{.dtype = DType::I8, .leading_extent = 128,
                                            .head_extent = 8},
                           PagedKVPlaneSpec{.dtype = DType::I8, .leading_extent = 128,
                                            .head_extent = 8},
                           PagedKVPlaneSpec{.dtype = DType::FP16, .leading_extent = 2,
                                            .head_extent = 8},
                           PagedKVPlaneSpec{.dtype = DType::FP16, .leading_extent = 2,
                                            .head_extent = 8}});
}

int test_layout_planning() {
    const PagedKVPoolSpec spec = quantized_text_spec();
    const HostKVPageLayout layout = plan_host_kv_page_layout(spec);

    if (layout.device_plane_order != PagedKVPlaneOrder::PageMajor) {
        failures += fail("layout plane order");
    }
    if (layout.planes.size() != 4) { failures += fail("layout plane count"); return failures; }

    // k plane: 128 * 64 * 1 (I8) = 8192 B per head, 8 heads = 65536 B per page.
    if (expect_eq(layout.planes[0].head_payload_bytes, 8192, "k head payload")) { return failures; }
    if (expect_eq(layout.planes[0].page_payload_bytes, 65536, "k page payload")) { return failures; }
    // scale plane: 2 * 64 * 2 (FP16) = 256 B per head, 8 heads = 2048 B per page.
    if (expect_eq(layout.planes[2].head_payload_bytes, 256, "scale head payload")) {
        return failures;
    }
    if (expect_eq(layout.planes[2].page_payload_bytes, 2048, "scale page payload")) {
        return failures;
    }

    std::size_t cursor = 0;
    for (const HostKVPlaneLayout& plane : layout.planes) {
        if (plane.offset != cursor) { failures += fail("plane offset packing"); return failures; }
        cursor += plane.page_payload_bytes;
    }
    // 65536 + 65536 + 2048 + 2048 = 135168, already 256-aligned.
    if (expect_eq(layout.page_stride, 135168, "page stride")) { return failures; }

    // A non-aligned payload must pad the record: one 100-byte plane rounds the stride to 256.
    const HostKVPageLayout padded =
        plan_host_kv_page_layout(make_pool_spec(
            PagedKVPlaneOrder::PageMajor,
            {PagedKVPlaneSpec{.dtype = DType::I8, .leading_extent = 1, .head_extent = 1}}));
    // 1 * 64 * 1 = 64 B payload -> 256 B record.
    if (expect_eq(padded.planes[0].page_payload_bytes, 64, "padded payload")) { return failures; }
    if (expect_eq(padded.page_stride, 256, "padded stride")) { return failures; }

    return expect_throws(
        [] {
            (void)plan_host_kv_page_layout(make_pool_spec(PagedKVPlaneOrder::PageMajor, {}));
        },
        "empty plane set");
}

int test_allocation_lifecycle() {
    const PagedKVPoolSpec spec = quantized_text_spec();
    const HostKVPageLayout layout = plan_host_kv_page_layout(spec);
    const std::size_t page_bytes = layout.page_stride; // 135168
    std::vector<HostKVPageLayout> layouts{layout};

    // 4 MiB holds 31 whole 135168-byte page records.
    HostKVArena arena(4ULL << 20, layouts);
    if (expect_eq(arena.capacity_bytes(), 4ULL << 20, "capacity")) { return failures; }
    if (expect(arena.occupied_bytes() == 0, "initially empty")) { return failures; }
    if (const HostKVPageLayout* found = arena.layout_for(spec); found == nullptr || *found != layout) {
        failures += fail("layout_for miss");
        return failures;
    }

    if (!arena.can_allocate(layout, 31)) { failures += fail("can_allocate full fit"); return failures; }
    if (arena.can_allocate(layout, 32)) { failures += fail("can_allocate over capacity"); return failures; }

    auto a = arena.allocate(layout, 16);
    auto b = arena.allocate(layout, 8);
    auto c = arena.allocate(layout, 6);
    if (!a || !b || !c) { failures += fail("allocation failed"); return failures; }
    if (expect_eq(arena.occupied_bytes(), (16 + 8 + 6) * page_bytes, "occupied after 3 allocs")) {
        return failures;
    }
    std::optional<HostKVAllocation> extra;
    // 30 records used: one more record (the 31st) still fits, the 32nd must not.
    if (!arena.allocate(layout, 1)) { failures += fail("capacity exhausted too early"); return failures; }
    extra = arena.allocate(layout, 1);
    if (!extra) { failures += fail("31st record refused"); return failures; }
    if (arena.allocate(layout, 1)) { failures += fail("capacity exceeded"); return failures; }
    extra.reset();

    // Views land on distinct 256-aligned records inside the backing.
    const HostKVAllocationView va = arena.view(*a);
    const HostKVAllocationView vb = arena.view(*b);
    const HostKVAllocationView vc = arena.view(*c);
    if (!va.valid() || !vb.valid() || !vc.valid()) { failures += fail("fresh views invalid"); return failures; }
    if (expect_eq(va.page_count(), 16, "view page count")) { return failures; }
    if (va.data() == vb.data() || vb.data() == vc.data()) {
        failures += fail("views alias");
        return failures;
    }
    const std::uintptr_t base = reinterpret_cast<std::uintptr_t>(va.data());
    if (base % 256 != 0) { failures += fail("record not 256-aligned"); return failures; }
    if ((va.data() + page_bytes * 16) > (vb.data()) || (vb.data() + page_bytes * 8) > (vc.data())) {
        failures += fail("first-fit ordering violated");
        return failures;
    }

    // Subview lands on the sub-run's first page record.
    const HostKVAllocationView vab = va.subview(4, 12);
    if (vab.data() != va.data() + page_bytes * 4) {
        failures += fail("subview base");
        return failures;
    }
    if (expect_eq(vab.page_count(), 12, "subview page count")) { return failures; }
    if (expect_throws([&] { (void)va.subview(16, 1); }, "subview past the end") != 0) { return failures; }

    // Any arena mutation invalidates outstanding views (revision bump); a view taken after
    // the mutation is fresh again.
    b.reset();
    if (va.valid() || vb.valid() || vc.valid()) { failures += fail("views survived release"); return failures; }
    if (!arena.view(*a).valid()) { failures += fail("fresh view after mutation"); return failures; }

    // The released middle extent (8 records) plus its neighbours must coalesce: after dropping
    // all three, a full-capacity allocation succeeds again.
    a.reset();
    c.reset();
    if (!arena.can_allocate(layout, 31)) { failures += fail("coalesce full capacity"); return failures; }
    if (expect_eq(arena.occupied_bytes(), 0, "occupied after releases")) { return failures; }

    // Move-only handle semantics.
    auto d = arena.allocate(layout, 4);
    if (!d) { failures += fail("move setup allocation"); return failures; }
    HostKVAllocation d2(std::move(*d));
    if (d->valid()) { failures += fail("move source armed"); return failures; }
    if (!d2.valid()) { failures += fail("move target disarmed"); return failures; }
    const HostKVAllocationView vd = arena.view(d2);
    d2.release();
    if (vd.valid()) { failures += fail("view survived explicit release"); return failures; }
    if (d2.valid()) { failures += fail("allocation survived release"); return failures; }

    // Reuse after release hands out a fresh generation; the old view must stay stale.
    auto e = arena.allocate(layout, 4);
    if (!e) { failures += fail("reuse allocation failed"); return failures; }
    e.reset();
    return 0;
}

int test_closed_layout_set() {
    const PagedKVPoolSpec text = quantized_text_spec();
    // The MTP backend pool at the same rank: fewer heads, 2 planes.
    const PagedKVPoolSpec backend = make_pool_spec(
        PagedKVPlaneOrder::PageMajor,
        {PagedKVPlaneSpec{.dtype = DType::I8, .leading_extent = 128, .head_extent = 4},
         PagedKVPlaneSpec{.dtype = DType::I8, .leading_extent = 128, .head_extent = 4}});
    const HostKVPageLayout text_layout = plan_host_kv_page_layout(text);
    const HostKVPageLayout backend_layout = plan_host_kv_page_layout(backend);
    std::vector<HostKVPageLayout> layouts{text_layout, backend_layout};
    HostKVArena arena(2ULL << 20, layouts);

    if (const HostKVPageLayout* found = arena.layout_for(text); found == nullptr || *found != text_layout) {
        failures += fail("text layout lookup");
        return failures;
    }
    if (const HostKVPageLayout* found = arena.layout_for(backend); found == nullptr ||
        *found != backend_layout) {
        failures += fail("backend layout lookup");
        return failures;
    }
    // A geometry not in the closed set is refused, not implicitly learned.
    const PagedKVPoolSpec other = make_pool_spec(
        PagedKVPlaneOrder::PageMajor,
        {PagedKVPlaneSpec{.dtype = DType::BF16, .leading_extent = 128, .head_extent = 8}});
    if (arena.layout_for(other) != nullptr) { failures += fail("closed set leaked"); return failures; }
    if (arena.allocate(backend_layout, 1) == std::nullopt) {
        failures += fail("backend allocation refused");
        return failures;
    }
    // Distinct layouts must not alias page records.
    auto t = arena.allocate(text_layout, 8);
    auto m = arena.allocate(backend_layout, 8);
    if (!t || !m) { failures += fail("mixed allocation failed"); return failures; }
    const HostKVAllocationView vt = arena.view(*t);
    const HostKVAllocationView vm = arena.view(*m);
    if (vt.data() == vm.data()) { failures += fail("mixed layouts alias"); return failures; }
    if (expect(vt.layout() == text_layout, "text view layout")) { return failures; }
    if (expect(vm.layout() == backend_layout, "backend view layout")) { return failures; }
    return 0;
}

int test_fragmentation() {
    const HostKVPageLayout layout = plan_host_kv_page_layout(quantized_text_spec());
    const std::size_t page_bytes = layout.page_stride;
    std::vector<HostKVPageLayout> layouts{layout};
    // Exactly 10 records.
    HostKVArena arena(10 * page_bytes, layouts);

    auto a = arena.allocate(layout, 2);
    auto b = arena.allocate(layout, 2);
    auto c = arena.allocate(layout, 2);
    auto d = arena.allocate(layout, 2);
    auto e = arena.allocate(layout, 2);
    if (!a || !b || !c || !d || !e) { failures += fail("fragmented setup failed"); return failures; }
    if (arena.allocate(layout, 1)) { failures += fail("capacity exceeded in setup"); return failures; }

    // Free records 0-1 and 2-3 -> they coalesce into one 4-record extent; a 4-record request
    // must now fit even though 4 single records are nominally free without coalescing.
    a.reset();
    b.reset();
    if (!arena.can_allocate(layout, 4)) { failures += fail("neighbour coalescing"); return failures; }
    auto big = arena.allocate(layout, 4);
    if (!big) { failures += fail("coalesced allocation failed"); return failures; }
    if (expect_eq(arena.free_bytes(), 0, "free after coalesced alloc")) { return failures; }
    c.reset();
    d.reset();
    e.reset();
    big.reset();
    if (!arena.can_allocate(layout, 10)) { failures += fail("full coalesce"); return failures; }
    return 0;
}

} // namespace

int test_sequential_feasibility() {
    const PagedKVPoolSpec text = quantized_text_spec();
    const PagedKVPoolSpec backend = make_pool_spec(
        PagedKVPlaneOrder::PageMajor,
        {PagedKVPlaneSpec{.dtype = DType::I8, .leading_extent = 128, .head_extent = 4},
         PagedKVPlaneSpec{.dtype = DType::I8, .leading_extent = 128, .head_extent = 4}});
    const HostKVPageLayout text_layout = plan_host_kv_page_layout(text);
    const HostKVPageLayout backend_layout = plan_host_kv_page_layout(backend);
    std::vector<HostKVPageLayout> layouts{text_layout, backend_layout};

    // One free extent in which each region alone fits but the regions combined exceed it:
    // the shape of the production "allocation failed after feasibility check" incident, where
    // per-region can_allocate() returned true yet the ordered allocation exhausted the extent
    // on the second region.
    const std::size_t capacity = 2 * text_layout.page_stride + backend_layout.page_stride;
    HostKVArena arena(capacity, layouts);

    const HostKVAllocationPlan text_region{&text_layout, 2};
    const HostKVAllocationPlan backend_region{&backend_layout, 3};
    const std::array<HostKVAllocationPlan, 2> plan{text_region, backend_region};
    if (!arena.can_allocate(text_layout, 2)) {
        failures += fail("text region alone should fit");
        return failures;
    }
    if (!arena.can_allocate(backend_layout, 3)) {
        failures += fail("backend region alone should fit");
        return failures;
    }
    if (arena.can_allocate_all(plan)) {
        failures += fail("sequential feasibility over-approximated");
        return failures;
    }
    // The simulation must agree with the real ordered allocation.
    auto t = arena.allocate(text_layout, 2);
    if (!t) { failures += fail("text allocation refused"); return failures; }
    if (arena.allocate(backend_layout, 3) != std::nullopt) {
        failures += fail("backend allocation should be exhausted");
        return failures;
    }
    t.reset();

    // Multi-extent first-fit: with two free extents the second region may land on the second
    // extent (success) or nowhere even though the total free bytes suffice (fragmentation).
    // Both outcomes must match the real ordered allocation exactly.
    const HostKVPageLayout single = plan_host_kv_page_layout(text);
    std::vector<HostKVPageLayout> one{single};
    const std::size_t stride = single.page_stride;
    // 10 records: a(2) active | b(3) freed | c(3) active | tail(2) -> two free extents of
    // 3S and 2S.
    HostKVArena multi(10 * stride, one);
    auto ma = multi.allocate(single, 2);
    auto mb = multi.allocate(single, 3);
    auto mc = multi.allocate(single, 3);
    if (!ma || !mb || !mc) { failures += fail("multi-extent setup refused"); return failures; }
    mb.reset();

    const HostKVAllocationPlan m1{&single, 3};
    const HostKVAllocationPlan m2{&single, 1};
    const std::array<HostKVAllocationPlan, 2> cross{m1, m2};
    // region 1 consumes the whole 3S extent; region 2 lands on the second 2S extent.
    if (!multi.can_allocate_all(cross)) {
        failures += fail("second-extent sequential plan refused");
        return failures;
    }
    auto mc1 = multi.allocate(single, 3);
    auto mc2 = multi.allocate(single, 1);
    if (!mc1 || !mc2) { failures += fail("second-extent ordered allocation refused"); return failures; }
    mc1.reset();
    mc2.reset();

    // Fragmentation: region 1 takes the front of the 3S extent leaving 2S, region 2 needs 3S;
    // the total free bytes (5S) cover the plan (4S) but no single extent holds it, so no
    // first-fit ordering places the second region.
    const HostKVAllocationPlan f1{&single, 1};
    const HostKVAllocationPlan f2{&single, 3};
    const std::array<HostKVAllocationPlan, 2> frag{f1, f2};
    if (multi.free_bytes() < 4 * stride) {
        failures += fail("fragmentation setup did not leave enough free bytes");
        return failures;
    }
    if (multi.can_allocate(single, f1.pages + f2.pages)) {
        failures += fail("fragmented arena reports a contiguous fit");
        return failures;
    }
    if (multi.can_allocate_all(frag)) {
        failures += fail("fragmented sequential plan over-approximated");
        return failures;
    }
    auto ff1 = multi.allocate(single, f1.pages);
    if (!ff1) { failures += fail("fragmentation region 1 refused"); return failures; }
    if (multi.allocate(single, f2.pages) != std::nullopt) {
        failures += fail("fragmentation region 2 should fail");
        return failures;
    }
    ff1.reset();

    // The inclusive boundary: a plan whose regions sum to exactly the free extent fits.
    const HostKVAllocationPlan exact_backend{&backend_layout, 1};
    const std::array<HostKVAllocationPlan, 2> exact{text_region, exact_backend};
    if (!arena.can_allocate_all(exact)) {
        failures += fail("exact-fit sequential plan refused");
        return failures;
    }
    auto e1 = arena.allocate(text_layout, 2);
    auto e2 = arena.allocate(backend_layout, 1);
    if (!e1 || !e2) {
        failures += fail("exact-fit ordered allocation refused");
        return failures;
    }
    if (arena.free_bytes() != 0) { failures += fail("exact fit left free bytes"); return failures; }
    return 0;
}

int test_raw_extents() {
    const PagedKVPoolSpec spec = quantized_text_spec();
    const HostKVPageLayout layout = plan_host_kv_page_layout(spec);
    const std::size_t page_bytes = layout.page_stride;
    std::vector<HostKVPageLayout> layouts{layout};
    HostKVArena arena(1ULL << 20, layouts);

    // A raw extent of an unaligned request size lands on 256-aligned backing and reports the
    // 256-rounded-up byte count.
    auto raw = arena.allocate_raw(1000);
    if (!raw) { failures += fail("raw allocation failed"); return failures; }
    if (expect_eq(raw->raw_bytes(), 1024, "raw byte count rounded to 256")) { return failures; }
    if (expect_eq(raw->page_count(), 0, "raw page count")) { return failures; }
    if (expect_eq(arena.occupied_bytes(), 1024, "occupied is exact bytes")) { return failures; }
    const HostKVAllocationView rv = arena.raw_view(*raw);
    if (!rv.valid() || rv.data() == nullptr) { failures += fail("raw view invalid"); return failures; }
    if (expect_eq(rv.raw_bytes(), 1024, "raw view byte count")) { return failures; }
    if (reinterpret_cast<std::uintptr_t>(rv.data()) % 256 != 0) {
        failures += fail("raw extent not 256-aligned");
        return failures;
    }

    // Raw and page runs share one budget and free-extent pool: the page run lands first-fit
    // after the raw extent, and the combined feasibility simulation agrees.
    auto page = arena.allocate(layout, 4);
    if (!page) { failures += fail("page allocation beside raw failed"); return failures; }
    if (expect_eq(arena.occupied_bytes(), 1024 + 4 * page_bytes, "combined occupied")) {
        return failures;
    }
    page.reset();
    if (!raw || !raw->raw_bytes()) { failures += fail("raw survived page release"); return failures; }
    if (!arena.raw_view(*raw).valid()) { failures += fail("raw view after page release"); return failures; }

    // Exhaustion: a raw request larger than the whole free span fails and leaves the arena
    // unchanged.
    const std::size_t free_now = arena.free_bytes();
    const std::size_t before = arena.occupied_bytes();
    if (arena.allocate_raw(free_now + 1)) {
        failures += fail("raw over capacity succeeded");
        return failures;
    }
    if (expect_eq(arena.occupied_bytes(), before, "failed raw leaves arena unchanged")) {
        return failures;
    }

    // raw_view on a page-run allocation throws; subview on a raw extent throws.
    auto page2 = arena.allocate(layout, 2);
    if (!page2) { failures += fail("page allocation for view checks"); return failures; }
    if (expect_throws([&] { (void)arena.raw_view(*page2); }, "raw_view of page run") != 0) {
        return failures;
    }
    if (expect_throws([&] { (void)rv.subview(0, 1); }, "subview of raw extent") != 0) {
        return failures;
    }

    // can_allocate_all with a mixed page/raw plan must match a same-order allocation run.
    raw.reset();
    page2.reset();
    const HostKVAllocationPlan over_page{&layout, 8};
    const HostKVAllocationPlan over_raw{nullptr, 0, 500000};
    const std::array<HostKVAllocationPlan, 2> mixed_over{over_page, over_raw};
    // 8 * 135168 + 500000 > 1 MiB -> must be refused.
    if (arena.can_allocate_all(mixed_over)) {
        failures += fail("mixed feasibility over capacity");
        return failures;
    }
    const HostKVAllocationPlan ok_page{&layout, 2};
    const HostKVAllocationPlan ok_raw{nullptr, 0, 200000};
    const std::array<HostKVAllocationPlan, 2> mixed_ok{ok_page, ok_raw};
    if (!arena.can_allocate_all(mixed_ok)) {
        failures += fail("mixed feasibility under capacity");
        return failures;
    }
    auto pa = arena.allocate(layout, 2);
    auto pb = arena.allocate_raw(200000);
    if (!pa || !pb) { failures += fail("mixed allocation run"); return failures; }
    pa.reset();
    pb.reset();
    if (expect_eq(arena.occupied_bytes(), 0, "all raw/page extents returned")) { return failures; }
    return 0;
}

int main() {
    // Each test function accumulates into the global `failures` and returns the running total;
    // calling them here without += avoids double-counting that total.
    (void)test_layout_planning();
    (void)test_allocation_lifecycle();
    (void)test_closed_layout_set();
    (void)test_fragmentation();
    (void)test_sequential_feasibility();
    (void)test_raw_extents();
    if (failures == 0) { std::cout << "ok\n"; return 0; }
    std::cerr << failures << " host KV arena check(s) failed\n";
    return 1;
}