# FINUFFT Issue Roadmap

**Updated:** 2026-10-21

> **Status: charted.** The wayfinder map for the cuFINUFFT Morton output-owned spread
> effort exists under `docs/issues/`. No sprint work items are planned until its go/no-go
> decision lands; the map and its tickets are planning artifacts, not sprint items.

## Purpose

This document is the ordered plan for the `docs/issues/` tracker. It sits above the issue
files: the roadmap decides *which* issues are created and in what order, while each issue
file carries the spec, design, implementation plan, and evidence for one unit of work.

The roadmap is not a second tracker. It does not restate issue details; it points at the
source of each work item and records only the ordering, dependencies, and decisions that
apply across issues.

## Current state

- The tracker layout exists: `docs/issues/todo/`, `docs/issues/in-progress/`, and
  `docs/issues/done/`.
- The wayfinder map [cuFINUFFT Morton output-owned spread go/no-go](issues/todo/ISSUE-001-cufinufft-morton-spread-go-no-go.md)
  is charted, with decision tickets covering research, a baseline task, grilling, and two
  prototype experiments.
- No implementation issues are filed and no sprints are planned. The next step is to work
  the map's decision tickets, starting with the unblocked baseline task and grilling
  tickets.
- The domain vocabulary is initialized in [`GLOSSARY.md`](GLOSSARY.md); the ownership term
  is pending [Settle the ownership and Morton vocabulary](issues/todo/ISSUE-007-ownership-and-morton-vocabulary.md).
- CPU perftest baseline timings for transform types 1-3 are recorded in the
  [appendix](#appendix-baseline-benchmarks); the map's CUDA baseline is a separate ticket.

## Destination

A measured go/no-go, on this machine's Quadro T1000, on whether Morton-sorted
output-cell-owned spreading beats cuFINUFFT `gpu_method=3` on 3D type-1 spreading at the
agreed success threshold, plus a settled integration design if the answer is go. The
[wayfinder map](issues/todo/ISSUE-001-cufinufft-morton-spread-go-no-go.md) holds the
destination, route, and open decisions.

## Sources and cross-reference

The map and its design sketch are the sources for the current effort; the appendix records
the CPU perftest baseline referenced in Current state.

| Source | Work items |
|---|---|
| [Wayfinder map](issues/todo/ISSUE-001-cufinufft-morton-spread-go-no-go.md) | The map's research, baseline, grilling, and prototype tickets |
| [Design sketch](issues/assets/ISSUE-001-destination-owned-nufft-design.md) | The two prototype experiments and the integration design |

## Priority mapping

`/issue-sprint-spec` uses this default unless the sprint table says otherwise:

| Priority | Sprints |
|---|---|
| P0 | Sprint 1 |
| P1 | Sprints 2–6 |
| P2 | Sprints 7–9 |
| P3 | Sprints 10–11 |

Priority reflects urgency and sequencing, not issue size. Size an issue for roughly
300–350 changed lines and split it into smaller vertical slices when it exceeds that.

## Sequencing principles

_To be settled with the map. Record only principles the planning session actually adopts.
Useful candidates include: correctness and measurement before optimization; one vertical
slice per issue; the build and full test suite stay green after every issue; interfaces
settle before implementation plans._

## Sprints

Each sprint ends when all of its issues are `Done` and the exit gate holds, not on a
calendar date. The status column should name the filed issue ids as they are created.

| Sprint | Status | Focus | Exit gate |
|---|---|---|---|
| _None yet_ | Not planned | _To be set by `wayfinder`_ | _To be set by `wayfinder`_ |

## Issue-creation order

_None yet. Once the map is charted, list the work items in dependency order here, with the
`ISSUE-NNN` id of any file that already exists. `/issue-sprint-spec` follows this order
and updates it as issues are filed._

## Decisions

_Decisions that affect ordering, scope, or priority live in the `wayfinder` map and its
tickets. Index the ones that constrain the roadmap here, with a link to the ticket that
holds the detail. Do not restate a decision in two places._

| Date | Decision | Recorded in |
|---|---|---|
| 2026-10-21 | The first effort is a go/no-go on Morton output-owned spreading for 3D type-1 cuFINUFFT; CPU, type 3, non-CUDA backends, and multi-GPU are out of scope. | [Wayfinder map](issues/todo/ISSUE-001-cufinufft-morton-spread-go-no-go.md) |

## Maintenance

- Keep this roadmap current as issues are specified, started, and closed; update it in the
  same session as the tracker change that made it stale.
- Change only what is true. Do not invent issue ids, sprint names, or completion evidence.
- When a `wayfinder` decision changes the plan, update the affected sprint row, the
  issue-creation order, and the decisions index. Leave the closed map and ticket bodies
  intact; append to them instead of rewriting them.
- When an issue is split, record the child ids and their blocking edges in the issue's
  Notes and in the roadmap's issue-creation order.

## Appendix: baseline benchmarks

_Measured 2026-10-21 on commit `ee272845`, before any performance work. These are the
reference numbers for optimization issues filed from this roadmap. Re-measure on the target
machine rather than comparing absolute times wholesale._

**Environment.** Intel Core i7-9850H (6 physical / 12 logical cores, 2.6-4.6 GHz; observed
2.8-4.1 GHz), Debian 12, Linux 6.8.0, 30 GiB RAM, GCC 12.2.0, CMake 3.25.1, Ninja. Release
build (`-O3 -march=native`), OpenMP on, FFTW 3.3.10 built from source by CMake/CPM (AVX2,
OpenMP), `FINUFFT_USE_DUCC0=OFF`. Shared container, so the machine is not quiet.

**Method.** `perftest/perftest` with `--sort=2 --upsampfact=0 --debug=0 --n_runs=5`. Case
sizes follow the `PARAM_LIST` cases in `perftest/perftest_config.py` (the CI set), except
that the 2D whole-node case (`M=3e5`) is omitted and a 3D `f32` case is added at
`tol=1e-4`, above the single-precision rounding floor because the CI list only carries a 3D
`f64` case. Each stage is the minimum of 5 runs inside one invocation and of two invocations
(the passes differed by at most 7.2% per stage). `makeplan` is one-time and reported
separately; `per-call` is `setpts + execute`. 1-thread rows are pinned with `taskset -c 2`;
12-thread rows use `--threads=0`, i.e. all 12 logical CPUs (6 physical cores with SMT).

Reproduce one case:

```bash
cmake -B build/default -DFINUFFT_BUILD_TESTS=ON -DBUILD_TESTING=ON
cmake --build build/default --target perftest -j
taskset -c 2 build/default/perftest/perftest \
  --prec=d --type=1 --N1=320 --N2=320 --M=1e7 --tol=1e-9 \
  --n_runs=5 --threads=1 --sort=2 --upsampfact=0 --debug=0
```

### Single thread

Times in ms; throughput from `M / per-call`.

| Dim | Prec | Type | N (modes/targets) | M (NU pts) | tol | makeplan | setpts | execute | per-call | Mpts/s |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|
| 1D | f32 | 1 | 1e4 | 1e7 | 2e-3 | 0.08 | 28.63 | 214.05 | 242.68 | 41.21 |
| 1D | f32 | 2 | 1e4 | 1e7 | 2e-3 | 0.08 | 5.30 | 89.15 | 94.45 | 105.87 |
| 1D | f32 | 3 | 1e4 | 1e7 | 2e-3 | 0.08 | 182.37 | 264.41 | 446.78 | 22.38 |
| 1D | f64 | 1 | 1e4 | 1e7 | 1e-9 | 0.08 | 39.98 | 294.68 | 334.67 | 29.88 |
| 1D | f64 | 2 | 1e4 | 1e7 | 1e-9 | 0.08 | 5.35 | 186.02 | 191.36 | 52.26 |
| 1D | f64 | 3 | 1e4 | 1e7 | 1e-9 | 0.08 | 311.79 | 369.15 | 680.94 | 14.69 |
| 2D | f32 | 1 | 320 x 320 | 1e7 | 1e-4 | 0.08 | 60.00 | 396.94 | 456.94 | 21.88 |
| 2D | f32 | 2 | 320 x 320 | 1e7 | 1e-4 | 0.08 | 58.72 | 481.82 | 540.54 | 18.50 |
| 2D | f32 | 3 | 320 x 320 | 1e7 | 1e-4 | 0.09 | 258.93 | 433.43 | 692.36 | 14.44 |
| 2D | f64 | 1 | 320 x 320 | 1e7 | 1e-9 | 0.09 | 181.20 | 584.22 | 765.42 | 13.06 |
| 2D | f64 | 2 | 320 x 320 | 1e7 | 1e-9 | 0.08 | 179.90 | 731.78 | 911.69 | 10.97 |
| 2D | f64 | 3 | 320 x 320 | 1e7 | 1e-9 | 0.08 | 589.23 | 693.68 | 1282.90 | 7.79 |
| 3D | f32 | 1 | 192 x 192 x 128 | 8e4 | 1e-4 | 0.04 | 0.56 | 451.18 | 451.74 | 0.18 |
| 3D | f32 | 2 | 192 x 192 x 128 | 8e4 | 1e-4 | 0.08 | 0.49 | 388.51 | 389.00 | 0.21 |
| 3D | f32 | 3 | 192 x 192 x 128 | 8e4 | 1e-4 | 0.09 | 390.78 | 1198.95 | 1589.73 | 0.05 |
| 3D | f64 | 1 | 192 x 192 x 128 | 8e4 | 1e-7 | 0.05 | 1.60 | 383.85 | 385.45 | 0.21 |
| 3D | f64 | 2 | 192 x 192 x 128 | 8e4 | 1e-7 | 0.08 | 1.65 | 296.65 | 298.30 | 0.27 |
| 3D | f64 | 3 | 192 x 192 x 128 | 8e4 | 1e-7 | 0.08 | 521.36 | 2608.26 | 3129.62 | 0.03 |

### All cores

Same cases with `--threads=0`; speedup is 1-thread per-call divided by 12-thread per-call.

| Dim | Prec | Type | 1 thread (ms) | 12 threads (ms) | Speedup |
|---|---|---|---:|---:|---:|
| 1D | f32 | 1 | 242.68 | 74.30 | 3.27x |
| 1D | f32 | 2 | 94.45 | 24.68 | 3.83x |
| 1D | f32 | 3 | 446.78 | 156.80 | 2.85x |
| 1D | f64 | 1 | 334.67 | 88.28 | 3.79x |
| 1D | f64 | 2 | 191.36 | 49.10 | 3.90x |
| 1D | f64 | 3 | 680.94 | 230.81 | 2.95x |
| 2D | f32 | 1 | 456.94 | 145.48 | 3.14x |
| 2D | f32 | 2 | 540.54 | 140.05 | 3.86x |
| 2D | f32 | 3 | 692.36 | 248.18 | 2.79x |
| 2D | f64 | 1 | 765.42 | 230.78 | 3.32x |
| 2D | f64 | 2 | 911.69 | 224.46 | 4.06x |
| 2D | f64 | 3 | 1282.90 | 432.06 | 2.97x |
| 3D | f32 | 1 | 451.74 | 186.50 | 2.42x |
| 3D | f32 | 2 | 389.00 | 142.86 | 2.72x |
| 3D | f32 | 3 | 1589.73 | 531.27 | 2.99x |
| 3D | f64 | 1 | 385.45 | 216.86 | 1.78x |
| 3D | f64 | 2 | 298.30 | 136.26 | 2.19x |
| 3D | f64 | 3 | 3129.62 | 1097.14 | 2.85x |

The `Mpts/s` column counts the `M` nonuniform points only. For type 3, `M` is the number of
sources and `N` the number of targets, so the second spread of `N` targets is not in the
throughput figure; type-3 throughput is not comparable with types 1 and 2. The 3D cases are
FFT-dominated at `M=8e4`, so their throughput columns describe the grid, not the spreader.
Only the FFTW backend and `ntransf=1` were measured.
