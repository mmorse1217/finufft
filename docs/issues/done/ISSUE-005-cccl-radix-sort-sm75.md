# ISSUE-005: Verify CCCL radix-sort support for Morton keys on sm_75

**Created:** 2026-10-21
**Priority:** P1
**Status:** Done
**Labels:** wayfinder:research
**Started:** 2026-10-21
**Completed:** 2026-10-21

**Related commits:**
- _None yet_

**Dependencies:**
- _None_

## Description

Determine what the CCCL version used by this build (CUDA 12.8 path, `CUDA12_CCCL_VERSION`
3.4.2 in the top-level CMake) provides for sorting `(uint64_t Morton key, uint32_t point
index)` pairs on sm_75. Cover:

- the exact CUB APIs available in this build (device radix sort, sort pairs, in-place versus
  temporary storage), their headers and namespaces, and whether they are header-only or
  need a link target;
- 64-bit key support and performance caveats on Turing/sm_75;
- temporary-storage sizing and how to query it without an unnecessary host synchronization;
- stream and allocator integration, and whether the sort is deterministic and stable for
  equal keys;
- whether `cub::DeviceRadixSort` can sort keys with a payload and what the payload limits
  are;
- alternatives if 64-bit sorting is slow: split keys, 32-bit keys if the grid bit budget
  allows, or a two-pass key/payload sort.

Cite installed header or documentation paths and, if practical, a tiny compile-only check
(do not add it to the library build). Write findings to
`docs/issues/research/ISSUE-005-cccl-radix-sort-sm75.md` and append a short `## Answer` to
this ticket.

## Notes

- 2026-10-21 20:14 — Ticket created by wayfinder charting.
- 2026-10-21 20:20 — Resolved by a research subagent; findings in `docs/issues/research/ISSUE-005-cccl-radix-sort-sm75.md`; ticket closed.

## Answer

- The CUDA 12.8 build will resolve CCCL **3.4.2** (CUB `300402`), not the toolkit's CCCL
  2.7.0: `cmake/setupCCCL.cmake` requests major 3, the installed 2.7 config is
  incompatible, so CPM will fetch `v3.4.2` (expected at `build/gpu/_deps/cccl-src`;
  absent until a GPU configure). The source-level findings and exact header citations are
  in [the research note](../research/ISSUE-005-cccl-radix-sort-sm75.md).
- Use `cub::DeviceRadixSort::SortPairs<uint64_t, uint32_t>` from
  `<cub/device/device_radix_sort.cuh>` (namespace `cub`). CCCL is header-only via the
  `CCCL::CCCL` INTERFACE target. Prefer the `cub::DoubleBuffer` overload with two key and
  two value buffers.
- 64-bit keys are supported directly; no decomposer is needed. On sm_75 CUB uses
  `Policy700` and the 8-bit onesweep path: 8 passes for a 64-bit key vs 4 for a 32-bit key.
  A full 64-bit pair sort measured ~2x slower than `end_bit = 30` on the T1000; use
  `end_bit = B` (the used Morton bits) first. A `uint32_t` key is only slightly faster once
  the pass counts match.
- Temporary storage uses the standard `nullptr` query: host-side only, no kernel launch
  and no stream synchronization. The pointer API needs `12N` plus a small lookback
  allocation (12.37 B/item at N = 1M for 8-byte keys/4-byte values); the `DoubleBuffer`
  API excludes the caller's `N`-sized alternate buffers (about 0.37 B/item at N = 1M).
  Query once per plan shape and reuse. The legacy API takes an explicit `cudaStream_t`;
  allocate temp with the plan's `ThrustAllocatorAsync`/`gpu_scratch` on the same stream.
- `DeviceRadixSort` is stable: equal Morton keys preserve input payload order, so run
  construction is deterministic for a fixed input ordering. Payload has no API-level cap;
  a 4-byte `uint32_t` is trivial and covers FINUFFT's current `int M`.
- Recommendation for ISSUE-010: `uint64_t` key, `end_bit = B`, `DoubleBuffer`, no two-pass
  split or 32-bit-key path initially. If `B <= 32`, a `uint32_t` key is an optional
  memory/speed tweak. The env-based API is 3.4.0+ and is not in the toolkit's CUB 2.7;
  since CMake only requires CCCL major 3, either use the legacy API or guard
  `CUB_VERSION >= 300400`.
