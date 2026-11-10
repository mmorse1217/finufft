# Atomic-free spread: warp-owned 8×8×8 regions with 16 cells per lane

> Status: Needs confirmation (performance acceptable; readability refactor scope open)
> Last updated: Turn 10 (code-quality refactor proposal)

## Outcome

_To be clarified._ Working hypothesis: remove the measured uniform-case bottleneck
(L1/TEX 96.32%, MIO throttle 42.34% of stall cycles, 1.25 shared loads per output
cell) by adopting exp_a's lane ownership: one warp owns one `8×8×8` warp region,
and each lane owns a `2×2×4 = 16`-cell lane cell group.

## Scope

### In scope

- [Open] Change lane cell ownership from `2×1×2` (4 cells) to `2×2×4` (16 cells).
- [Open] Make one warp own one `8×8×8` warp region.
- [Open] Choose threadblock-shared vs per-warp staging and synchronization.
- [Open] Preserve the current zero-filled staged kernel-weight representation unless
  decided otherwise.
- [Open] Keep the standalone prototype readable and buildable via `run_atomic_free_spread.sh`.

### Non-goals

- [Open] Replacing 8 zero-filled weights per dimension with 6 Horner coefficients
  plus support placement.
- [Open] x-run batching across Cartesian bins.
- [Open] Split counts, active-region lists, partial grids, split reduction.
- [Open] GPU-side preprocessing/sorting.
- [Open] Store coalescing, output-layout, or bank-conflict work beyond keeping the
  new geometry conflict-free.

## Users and workflows

[Open] Primary user: the developer iterating on the atomic-free prototype. Secondary
user: a future implementation session that must execute this plan without guessing.

## Requirements and acceptance criteria

[Open] Candidate acceptance criteria, to be confirmed:

- Uniform spread median improves materially versus the current 19.471 ms.
- Uniform LDS instruction count drops by roughly 2.5× (128.5M → ~51M).
- No numerical regression against the bench reference.
- No severe regression on clustered, sparse, or dense.
- No local-memory spills; register count stays in the 85–102 range if possible.

## Constraints

- Target GPU from the latest capture: Quadro T1000, sm_75, 32 warps/SM,
  65536 registers/SM, 65.54 KB shared/SM, 16 blocks/SM.
- Current kernel: 128 threads/block, 512 blocks for `64³`, 63 registers/thread,
  4.64 KB static shared, 8 blocks/SM, 92.44% achieved occupancy.
- exp_a v11ay reference: 95–96 registers/thread, ~12.7 KB shared, 5 blocks/SM,
  10.912 ms uniform.
- No atomics: every output cell is written exactly once by one lane.
- The prototype is not part of the library build; changes stay under
  `devel/morton_spread/`.
- Readability is a stated goal; do not fragment the prototype into many shallow
  headers.

## Terminology (grounded in `docs/GLOSSARY.md`)

The earlier chat design did not consistently follow `docs/GLOSSARY.md`. It used
"batch" for a group of staged sources, "region" for the warp-owned output box,
and "halo" for the per-side support expansion. All three conflict with or are
undefined in the glossary. The corrected hierarchy is:

| level | term | status | definition | size in this design |
|---|---|---|---|---|
| launch unit | **CUDA threadblock** | glossary | group of threads launched together; shares shared memory and barriers | 4 warps, 128 threads |
| threadblock output region | **tile** | glossary | the output region a CUDA threadblock works on | e.g. `32×8×8` = four warp regions along x |
| SIMT execution unit | **warp** | local | 32 CUDA threads | 32 lanes |
| warp output region | **warp region** | local | an `8×8×8` subdivision of the tile owned by one warp | 512 cells |
| thread within a warp | **lane** | local | one of the 32 threads of a warp; used only for warp-relative identity | 1 of 32 |
| lane output cells | **lane cell group** | local | the `2×2×4 = 16` cells written by one lane | 16 cells |
| point grouping | **bin** | glossary | a box of fine-grid cells used to group sources | `bin_size=4` |
| bins that can reach a region | **candidate bins** | local | per-dimension bin ranges expanded by the support overhang | ~64 per warp region |
| staged sources | **staging group** | local | up to `STAGING_GROUP_SIZE` sorted sources whose kernel weights are staged together | 16 first |
| support beyond a region | **support overhang** | local | `ceil(ns/2)` cells per side; avoids the glossary-forbidden "halo" | 3 cells per side |
| source coefficient | **strength** | glossary | the complex coefficient attached to a source | `Point::strength` |
| spreading-kernel value | **kernel weight** | glossary | the spreading kernel's value at a fine-grid cell | staged in shared memory |

Important corrections:

- **"Batch" is not the staged source group.** The glossary reserves **batch** for
  a group of stacked transforms processed together in one FFT. The staged source
  group is a **staging group**. Rename `STAGE_POINTS` / `accumulate_batch`
  accordingly during implementation.
- **"Tile" is not the warp-owned `8³` box.** The tile is the CUDA threadblock's
  output region; the warp-owned `8³` box is a **warp region**. Never use "region"
  alone.
- **"Halo" is avoided.** Use **kernel support** for the `ns` cells and **support
  overhang** for the `ceil(ns/2)` per-side expansion. The code's `HALO` identifier
  should become `SUPPORT_OVERHANG`.
- **"Lane" vs "thread":** use **thread** for generic CUDA execution and launch
  geometry; use **lane** only when warp-relative identity matters, especially for
  the `2×2×4` lane cell group. Define the term once and do not mix them.

## Current-state facts and dependencies

### Benchmarks

| workload | current AF spread | exp_a v11ay |
|---|---:|---:|
| uniform `1e6`, `64³` | 19.471 ms | 10.912 ms |
| clustered `1e6`, `64³` | 26.782 ms | 11.232 ms |
| sparse `1e5`, `128³` | 5.968 ms | 1.393 ms |
| dense cluster `1e5`, `64³` | 8.326 ms | n/a |

### Uniform profile facts (latest capture)

- 734,451,480 warp instructions.
- L1/TEX throughput 96.32%; DRAM 5.40%.
- MIO throttle 42.34% of stall cycles; barrier 24.62%.
- LDS instructions 17.49% of total; 128.46M LDS instructions;
  160.53M shared-load wavefronts.
- FFMA 40.28% of instructions; FMUL 9.40%.
- 63 registers/thread; 4.64 KB static shared; no spills.

### Geometry facts

- Current: the CUDA threadblock owns an `8×8×8` tile; 4 warps cooperate; each
  thread (lane) owns a `2×1×2 = 4`-cell thread cell group; per source per thread:
  `2 kx + 1 ky + 2 kz = 5` LDS; `5/4 = 1.25` LDS/cell.
- Target: the CUDA threadblock owns a tile composed of four `8×8×8` warp regions;
  each warp owns one warp region; each lane owns a `2×2×4 = 16`-cell lane cell
  group; per source per lane: `2 kx + 2 ky + 4 kz = 8` LDS; `8/16 = 0.5` LDS/cell.
- FFMA per cell is unchanged at 2 (real + imaginary).
- Per-warp staged point size: `9 floats kx + 9 floats ky + 9 float2 kz = 144 bytes`.
- 4 warps, single buffer, `BATCH=16`: 9.0 KB shared/block.
- 4 warps, double buffer, `BATCH=11`: 12.4 KB shared/block; `BATCH=13`: 14.6 KB.

### Options under discussion

- **Option A:** threadblock-shared staging over a `16×16×8` tile of four warp
  regions; each warp owns an `8³` warp region; threadblock barriers remain.
  Staging is cheaper per cell, but each warp scans ~144 candidate bins instead of
  ~64.
- **Option B (current recommendation):** four independent warps per CUDA
  threadblock, each with its own warp region, source-range cursor, staging group,
  and `__syncwarp`.
- **Option C:** one warp per CUDA threadblock; the tile and the warp region are
  both `8³`. Simple, but limited to 16 warps/SM by the 16-threadblock/SM limit.
- **Option D:** threadblock-shared staging only for clustered, after splits exist.

## Tile growth: what it does and does not solve

The user approved growing the threadblock tile, but tile growth is not itself a
performance mechanism.

- **What it solves:** with 16 cells per lane, four warps need 2048 cells. A tile
  of four `8³` warp regions (e.g. `32×8×8`) lets the CUDA threadblock keep four
  warps, so the occupancy ceiling is set by registers/shared memory rather than by
  the 16-threadblock/SM limit. Option C (one warp per `8³` threadblock) caps at
  16 warps/SM; four warps per `32×8×8` threadblock reaches ~20 warps/SM at ~90
  registers.
- **What it does not solve:** the uniform bottleneck is 1.25 LDS/cell and four
  warps redundantly scanning the same tile sources. Tile growth changes neither.
  The LDS reduction comes from the `2×2×4` lane cell group; the redundant-scan
  reduction comes from one warp owning the whole warp region and enumerating its
  own candidate bins.
- **The condition that makes it safe:** the tile must remain a launch container
  only. Candidate-bin enumeration and staging must be per warp region. If the
  larger tile leaks into tile-wide candidate bins or threadblock-shared staging,
  the point reread factor and scan work grow, and tile growth becomes a regression
  (this is Option A's failure mode).
- **Residual cost:** a four-warp-region tile is coarser for future active-tile
  lists and splits. One active warp region makes the whole CUDA threadblock
  resident while up to three warps may be idle. Two warp regions per threadblock
  (`16×8×8`) is a middle ground.

## Decisions

- [Decided] Option B: independent warp per warp region, per-warp staging,
  `__syncwarp` instead of `__syncthreads`.
- [Decided] Keep zero-filled staged kernel weights for this change;
  coefficient staging is the next feature.
- [Decided] Single staging buffer, `STAGING_GROUP_SIZE=16`, first.
- [Decided] The CUDA threadblock tile may grow beyond `8³`. Tile growth is a
  packaging choice, not a performance mechanism.
- [Decided for the first implementation] Two warp regions per CUDA threadblock
  (`16×8×8` tile, 64 threads). Four warp regions (`32×8×8`) remains the
  exp_a-parity comparison via the compile-time constant.
- [Decided] The tile is a launch container only: candidate-bin enumeration and
  staging are per warp region. Tile size must not leak into either.
- [Decided] The existing bench correctness matrix is the hard gate. Host geometry
  and cursor tests are a follow-up, not a blocker for this prototype.
- [Decided] Preserve `docs/GLOSSARY.md`: **tile** remains the CUDA
  threadblock's output region. Introduce these local terms: **warp**, **lane**,
  **warp region**, **lane cell group**, **staging group**, **candidate bins**,
  **support overhang**.
- [Decided] Use **thread** for generic CUDA execution and launch geometry;
  use **lane** only for warp-relative identity and the `2×2×4` cell group.
- [Decided] Rename the point-staging use of "batch" to **staging group**.
  `docs/GLOSSARY.md` reserves **batch** for stacked transforms.
- [Decided] Keep the combined coefficient-staging + active-region/split kernel as
  the new best atomic-free baseline. It beats the old baseline on all four cases
  (uniform 16.70 vs 19.47 ms, clustered 16.43 vs 26.78 ms, sparse 2.24 vs
  5.97 ms, dense 1.94 vs 8.33 ms). It remains 1.46–1.61× behind exp_a on
  uniform/clustered/sparse.
- [Open] Next optimization: persistent workspace/compact partial grids to remove
  allocation outliers, or x-run batching to raise the non-wrapping support hit
  rate.

## Proposed design / plan

Module decomposition and implementation sketch:

| module | interface | hides |
|---|---|---|
| `WarpRegion` / `LaneCellGroup` | `warp_region_for(...)`, `lane_cell_group(warp_region, lane)` | lane bit mapping, extent, clipping |
| `KernelSupport` | `cell_weight(cell, start, grid_size, ker)` | periodic support index, out-of-grid zeros |
| `StagingGroup` | `stage_group(staging_group, sources, count, warp_region, grid, lane)` | shared layout, row padding, strength scaling |
| `CellAccumulator` | `add_staging_group(acc, staging_group, count, cell_group)`, `store(...)` | 2×2×4 unroll, kernel-weight reuse, store predicate |
| `SourceRangeCursor` | `bool next(int& begin, int& end)` | candidate-bin loops, periodic wrap; later x-runs |
| kernel | threadblock tile → warp region → lane cell group → cursor → staging group → accumulator → store | — |

### Header constants and public interface

```cpp
constexpr int WIDTH = 6;
constexpr int SUPPORT_OVERHANG = (WIDTH + 1) / 2;  // 3; replaces HALO

constexpr int WARP_REGION_X = 8;
constexpr int WARP_REGION_Y = 8;
constexpr int WARP_REGION_Z = 8;

constexpr int LANE_CELLS_X = 2;
constexpr int LANE_CELLS_Y = 2;
constexpr int LANE_CELLS_Z = 4;

constexpr int LANES_X = WARP_REGION_X / LANE_CELLS_X;  // 4
constexpr int LANES_Y = WARP_REGION_Y / LANE_CELLS_Y;  // 4
constexpr int LANES_Z = WARP_REGION_Z / LANE_CELLS_Z;  // 2
constexpr int LANES_PER_WARP = 32;

constexpr int WARP_REGIONS_PER_THREADBLOCK = 2;  // launch-policy knob: 1, 2, 4
constexpr int WARPS_PER_THREADBLOCK = WARP_REGIONS_PER_THREADBLOCK;
constexpr int THREADS_PER_THREADBLOCK = WARPS_PER_THREADBLOCK * 32;

constexpr int TILE_X = WARP_REGION_X * WARP_REGIONS_PER_THREADBLOCK;
constexpr int TILE_Y = WARP_REGION_Y;
constexpr int TILE_Z = WARP_REGION_Z;

constexpr int STAGING_GROUP_SIZE = 16;

static_assert(WARP_REGION_X % LANE_CELLS_X == 0);
static_assert(WARP_REGION_Y % LANE_CELLS_Y == 0);
static_assert(WARP_REGION_Z % LANE_CELLS_Z == 0);
static_assert(LANES_X * LANES_Y * LANES_Z == LANES_PER_WARP);
static_assert(WARP_REGION_X * WARP_REGION_Y * WARP_REGION_Z ==
              LANES_PER_WARP * LANE_CELLS_X * LANE_CELLS_Y * LANE_CELLS_Z);
static_assert(STAGING_GROUP_SIZE <= 32);  // one lane per source
```

The public `Point`, `Grid`, `launch_assign_bins`, and `launch_spread` signatures
stay unchanged.

### Device types and function signatures

```cpp
struct WarpRegion {
  int x, y, z;       // global origin
  int nx, ny, nz;    // clipped extent, 1..WARP_REGION_*
};

struct LaneCellGroup {
  int x, y, z;       // warp-region-local origin
};

struct Support {
  int start;
  float offset;      // start - fold_rescale(coordinate, grid_size)
};

struct CandidateBins {
  int first, count, bins;
  __device__ int operator[](int i) const;  // periodic wrap
};

struct StagingGroup {
  float kx[STAGING_GROUP_SIZE][WARP_REGION_X + 1];
  float ky[STAGING_GROUP_SIZE][WARP_REGION_Y + 1];
  float2 kz[STAGING_GROUP_SIZE][WARP_REGION_Z + 1];
};

struct CellAccumulator {
  float2 cell[LANE_CELLS_X][LANE_CELLS_Y][LANE_CELLS_Z];
};

__device__ WarpRegion make_warp_region(int x, int y, int z, Grid grid);
__device__ LaneCellGroup lane_cell_group(int lane);
__device__ Support make_support(float coordinate, int grid_size);

__device__ CandidateBins candidate_bins(int begin, int end, int bin_size,
                                        int grid_size, int bins);

class SourceRangeCursor {
 public:
  __device__ SourceRangeCursor(WarpRegion region, Grid grid,
                               const int* bin_offsets);
  __device__ bool next(int& begin, int& end);
};

template <int CELLS>
__device__ void stage_axis(float* dest, int region_begin, int grid_size,
                           float coordinate);

template <int CELLS>
__device__ void stage_axis(float2* dest, int region_begin, int grid_size,
                           float coordinate, float2 strength);

__device__ void stage_staging_group(StagingGroup& group, const Point* sources,
                                    int count, WarpRegion region, Grid grid,
                                    int lane);

__device__ void add_staging_group(CellAccumulator& acc,
                                  const StagingGroup& group, int count,
                                  LaneCellGroup cell_group);

__device__ void store_cell_group(const CellAccumulator& acc,
                                 cuFloatComplex* output, Grid grid,
                                 WarpRegion region, LaneCellGroup cell_group);
```

### Kernel pseudocode

```cpp
__global__ void spread(const Point* sources, const int* bin_offsets, Grid grid,
                       cuFloatComplex* output) {
  __shared__ StagingGroup staging[WARPS_PER_THREADBLOCK];

  const int warp = threadIdx.x / 32;
  const int lane = threadIdx.x % 32;
  const int region_x = blockIdx.x * TILE_X + warp * WARP_REGION_X;
  const int region_y = blockIdx.y * TILE_Y;
  const int region_z = blockIdx.z * TILE_Z;

  const WarpRegion region = make_warp_region(region_x, region_y, region_z, grid);
  if (region.nx <= 0 || region.ny <= 0 || region.nz <= 0) return;

  const LaneCellGroup cell_group = lane_cell_group(lane);
  CellAccumulator acc = {};
  StagingGroup& group = staging[warp];

  SourceRangeCursor cursor(region, grid, bin_offsets);
  int begin, end;
  while (cursor.next(begin, end)) {
    for (int base = begin; base < end; base += STAGING_GROUP_SIZE) {
      const int count = min(STAGING_GROUP_SIZE, end - base);
      stage_staging_group(group, sources + base, count, region, grid, lane);
      __syncwarp();
      add_staging_group(acc, group, count, cell_group);
      __syncwarp();
    }
  }

  store_cell_group(acc, output, grid, region, cell_group);
}
```

### Launch pseudocode

```cpp
void launch_spread(const Point* sources, const int* bin_offsets, Grid grid,
                   cuFloatComplex* output) {
  const dim3 threads(THREADS_PER_THREADBLOCK);
  const dim3 blocks((grid.nx + TILE_X - 1) / TILE_X,
                    (grid.ny + TILE_Y - 1) / TILE_Y,
                    (grid.nz + TILE_Z - 1) / TILE_Z);
  spread<<<blocks, threads>>>(sources, bin_offsets, grid, output);
}
```

### Helper pseudocode

`lane_cell_group`:

```cpp
const int lx = lane % LANES_X;
const int ly = (lane / LANES_X) % LANES_Y;
const int lz = lane / (LANES_X * LANES_Y);
return {lx * LANE_CELLS_X, ly * LANE_CELLS_Y, lz * LANE_CELLS_Z};
```

`stage_axis`:

```cpp
const Support support = make_support(coordinate, grid_size);
float ker[WIDTH];
eval_kernel_vec_horner<float, WIDTH>(ker, support.offset, 2.0);
#pragma unroll
for (int local = 0; local < CELLS; ++local)
  dest[local] = cell_weight(region_begin + local, support.start, grid_size, ker);
```

The `float2` overload is the same, but writes `make_float2(weight * strength.x,
weight * strength.y)`.

`add_staging_group`:

```cpp
for (int j = 0; j < count; ++j) {
  float xy[LANE_CELLS_X][LANE_CELLS_Y];
  #pragma unroll
  for (int lx = 0; lx < LANE_CELLS_X; ++lx)
    #pragma unroll
    for (int ly = 0; ly < LANE_CELLS_Y; ++ly)
      xy[lx][ly] = group.kx[j][cell_group.x + lx] *
                   group.ky[j][cell_group.y + ly];

  float2 z[LANE_CELLS_Z];
  #pragma unroll
  for (int lz = 0; lz < LANE_CELLS_Z; ++lz)
    z[lz] = group.kz[j][cell_group.z + lz];

  #pragma unroll
  for (int lx = 0; lx < LANE_CELLS_X; ++lx)
    #pragma unroll
    for (int ly = 0; ly < LANE_CELLS_Y; ++ly)
      #pragma unroll
      for (int lz = 0; lz < LANE_CELLS_Z; ++lz) {
        acc.cell[lx][ly][lz].x += xy[lx][ly] * z[lz].x;
        acc.cell[lx][ly][lz].y += xy[lx][ly] * z[lz].y;
      }
}
```

`stage_staging_group`:

```cpp
if (lane >= count) return;
const Point p = sources[lane];
stage_axis<WARP_REGION_X>(group.kx[lane], region.x, grid.nx, p.x);
stage_axis<WARP_REGION_Y>(group.ky[lane], region.y, grid.ny, p.y);
stage_axis<WARP_REGION_Z>(group.kz[lane], region.z, grid.nz, p.z,
                          make_float2(p.strength.x, p.strength.y));
```

The early return is from the helper, not the kernel; all lanes still reach the
kernel's `__syncwarp()`.

### What changes

| file | change |
|---|---|
| `atomic_free_spread.cuh` | Replace `HALO` with `SUPPORT_OVERHANG`; replace `TILE_*`, `GROUP_*`, `BLOCK_*`, `BLOCK_THREADS`, `STAGE_POINTS` with the new geometry and staging constants; add static asserts; update the header comment. Keep `Point`, `Grid`, and launch signatures. |
| `atomic_free_spread.cu` | Keep `assign_bins`, `floor_div`, `extended_bin_index`, `coefficient_index`, `cell_weight`. Rename `NearBins`/`near_bins` to `CandidateBins`/`candidate_bins`. Add `WarpRegion`, `LaneCellGroup`, `Support`, `make_support`, `make_warp_region`, `lane_cell_group`, `SourceRangeCursor`, `StagingGroup`, `CellAccumulator`. Replace `stage_tile_weights`, `accumulate_batch`, `store_cell_group`, the `spread` body, and the launch geometry. |
| `atomic_free_spread_bench.cu` | No expected change; it calls `launch_spread`. |
| `profile_atomic_free_spread.py` | No required change, but check the `SOURCE_GROUPS` regexes after the scan body changes. |
| `README.md` / `profile.md` | Update geometry, baseline numbers, and terminology after the first capture. |
| `docs/GLOSSARY.md` | Optional: promote `warp`, `lane`, `warp region`, and `staging group` if they become project-wide terms. |

### What stays unchanged

- `assign_bins`, preprocessing/sort, and the bench/reference harness.
- `fold_rescale`, `eval_kernel_vec_horner`, zero-filled kernel weights, and
  strength scaling.
- Output-cell ownership and the once-per-cell store rule.
- The public `Point`, `Grid`, and launch signatures.

### Sketch invariants

- The tile is a launch container only; `SourceRangeCursor` and `StagingGroup` are
  per warp region.
- One lane per source in a staging group; `STAGING_GROUP_SIZE <= 32`.
- `STAGING_GROUP_SIZE = 16` keeps the `9`-`float2` `kz` rows conflict-free for 16
  staging lanes; above 16 the row stride repeats banks.
- The staging group never crosses a candidate bin in this version. x-run batching
  is a later change inside `SourceRangeCursor`.
- The accumulator is indexed only by compile-time unrolled loops.
- There is no `__syncthreads()` in the scan. The only block-wide operation is the
  static shared-memory allocation.
- The early return for an out-of-grid warp region is warp-uniform; revisit it if a
  future block-wide barrier is added.

## Assumptions

- [Open] The uniform gain is primarily from LDS/cell reduction, not occupancy.
- [Open] Sixteen independent accumulators provide enough ILP to compensate for the
  4× reduction in resident warps.
- [Open] Warp-region-local candidate-bin enumeration (`~64` bins) is preferable
  to threadblock-wide `16×16×8` staging (`~144` bins) for uniform.
- [Open] Tile size does not affect scan work in Option B because each warp region
  enumerates its own candidate bins.
- [Open] `docs/GLOSSARY.md` remains authoritative; new terms stay local to this
  plan unless they are promoted.
- [Open] The T1000 is the only target GPU.

## Risks and failure modes

- [Open] Dense may regress because total warps drop 4× and dense is already
  latency-bound (16.45% achieved occupancy).
- [Open] Growing the threadblock tile from `8³` to four warp regions (`32×8×8`)
  changes the meaning of `TILE_*` in the code; `WARP_REGION_*` must be introduced
  so the glossary's tile/warp-region distinction is not lost.
- [Open] Future arbitrary active-region lists may give a CUDA threadblock four
  non-contiguous warp regions, which is not a single contiguous tile under the
  glossary definition. That future feature will need a new term or a different
  launch mapping.
- [Open] Per-warp shared staging multiplies shared memory by `WARPS_PER_BLOCK`;
  double buffering can push the kernel from 5 to 4 blocks/SM.
- [Open] Per-warp staging is less parallel than the current 96-lane block staging
  and may serialize.
- [Open] 16 `float2` accumulators may push registers past 102 and cause spills or
  a 4-block occupancy cap.
- [Open] If the larger tile leaks into threadblock-wide candidate bins or staging,
  the point reread factor and scan work grow; this is Option A's failure mode.
- [Open] A four-warp-region tile is coarser for future active-tile lists/splits:
  one active warp region can hold a whole CUDA threadblock resident while up to
  three warps idle.
- [Open] `PointRangeCursor` periodic-wrap logic and `NearBins` cap are easy to get
  wrong; duplicates or missed points are correctness bugs.
- [Open] Partial edge regions, out-of-grid weights, and store predication.
- [Open] `__syncwarp` reachability if a lane or warp returns early.
- [Open] Profiler source-group regexes may misclassify the new scan body.

## Validation, rollout, and operations

- [Open] Host unit tests: lane-cell-group bijection, edge-warp-region coverage,
  candidate-bin coverage under wrap, support placement versus brute force.
- [Open] End-to-end bench/reference comparison on uniform, clustered, sparse, dense.
- [Open] Nsight Compute capture on the same four cases; compare LDS, shared
  wavefronts, MIO/barrier stalls, registers, spills, bank conflicts.
- [Open] Rollback: keep the current kernel available until the new one passes the
  reference check and the uniform benchmark.

## Open questions

1. Keep the warp-region prototype as an experimental branch, or revert to the
   previous block-owned kernel while the next feature is designed?
2. If iterating, which comes first: coefficient staging (6 Horner coefficients +
   support placement) or a cheaper staging-parallelism experiment
   (`STAGING_GROUP_SIZE=32` or four warp regions per threadblock)?
3. Are splits/active-region lists the next feature for clustered/dense, or are
   those cases allowed to remain regressed while uniform/sparse are fixed?
4. For any next iteration, should the launch policy stay at two warp regions per
   CUDA threadblock or move to four for exp_a parity?
5. Is a double staging buffer still worth trying, or is single-buffer
   `__syncwarp` cheap enough that the staging representation dominates?
6. Should **warp**, **lane**, **warp region**, and **staging group** be promoted
   into `docs/GLOSSARY.md`, or remain local to this plan?

## Readiness assessment

**Verdict: executable as a prototype with defaults.** The remaining open items are
tuning knobs and acceptance thresholds, not architectural blockers. A competent
implementer can start without guessing if the defaults below are accepted.

Defaults to implement unless overridden:

- Option B: one independent warp per `8³` warp region; per-warp candidate-bin
  cursor, staging group, and `__syncwarp`.
- `WARP_REGIONS_PER_THREADBLOCK` is a compile-time launch-policy parameter. The
  first measurement uses 2 (tile `16×8×8`); 4 is the exp_a-parity comparison.
- Single staging buffer, `STAGING_GROUP_SIZE=16`, zero-filled staged kernel
  weights.
- 3D launch grid; warp regions packed along x within the tile.
- Host unit tests for lane-cell-group bijection, edge warp regions, candidate-bin
  coverage, and support placement; the existing bench/reference covers numerics.

Still open but **non-blocking** for a prototype:

- numeric success threshold for uniform (set after the first measurement),
- acceptable dense/clustered/sparse regression,
- final 2-vs-4 warp regions per threadblock,
- single vs double staging buffer,
- whether to promote local terms into `docs/GLOSSARY.md`.

## Profiling result (warp-region prototype)

Capture: `build/gpu/traces/af_profile_20261110_170201/`
Command: `./devel/morton_spread/profile_atomic_free_spread.py`
Kernel: `WARP_REGIONS_PER_THREADBLOCK=2`, 89 registers, 4.61 KB static shared,
no spills. All four cases pass the bench correctness checks.

### Benchmark and correctness

| case | new median | old median | delta | exp_a | new / exp_a | relL2 |
|---|---:|---:|---:|---:|---:|---:|
| uniform | **18.451 ms** | 19.471 ms | **−5.2%** | 10.912 ms | 1.69× | 5.285e−07 |
| clustered | **61.145 ms** | 26.782 ms | **+128.3%** | 11.232 ms | 5.44× | 6.469e−06 |
| sparse | **4.635 ms** | 5.968 ms | **−22.3%** | 1.393 ms | 3.33× | 7.964e−08 |
| dense | **20.676 ms** | 8.326 ms | **+148.3%** | n/a | n/a | 8.388e−06 |

### Uniform: old → new

| metric | old | new | change |
|---|---:|---:|---:|
| ncu duration | 19.51 ms | 18.40 ms | −5.7% |
| warp instructions | 734,451,480 | 854,611,148 | **+16.4%** |
| LDS instructions | 128,460,416 | 64,000,000 | **−50.2%** |
| L1/TEX throughput | 96.32% | 46.19% | −50.1 pp |
| MIO throttle share | 42.34% | 0.33% | −42.0 pp |
| barrier share | 24.62% | 0.00% | −24.6 pp |
| FMA pipe | 56.41% | 76.03% | +19.6 pp |
| math-pipe throttle share | 3.66% | 21.78% | +18.1 pp |
| issue slots busy | 50.49% | 63.30% | +12.8 pp |
| IPC | 2.02 | 2.53 | +25% |
| achieved occupancy | 92.44% | 48.62% | −43.8 pp |
| no-eligible warps | 49.52% | 33.45% | −16.1 pp |
| registers/thread | 63 | 89 | +26 |
| waves/SM | 4.57 | 1.83 | −60% |

### Cross-case key metrics (new capture)

| metric | uniform | clustered | sparse | dense |
|---|---:|---:|---:|---:|
| ncu duration (ms) | 18.40 | 61.09 | 4.64 | 20.79 |
| compute (SM) % | 73.75 | 21.81 | 74.27 | 6.42 |
| memory % | 44.80 | 13.54 | 31.47 | 4.03 |
| L1/TEX % | 46.19 | 23.76 | 32.96 | 9.29 |
| IPC | 2.53 | 1.26 | 2.98 | 0.49 |
| issue slots busy % | 63.30 | 31.49 | 74.56 | 12.21 |
| FMA pipe % | 76.03 | 38.28 | 70.95 | 14.79 |
| ALU pipe % | 27.98 | 13.39 | 51.06 | 5.25 |
| achieved occupancy % | 48.62 | 14.88 | 57.57 | 4.29 |
| active warps/scheduler | 4.04 | 1.95 | 4.54 | 1.05 |
| no eligible % | 33.45 | 50.26 | 25.04 | 62.96 |
| active threads/warp | 23.21 | 24.12 | 9.96 | 24.67 |
| warp instructions | 854.6M | 821.8M | 258.9M | 82.3M |
| registers/thread | 89 | 89 | 89 | 89 |
| waves/SM | 1.83 | 1.83 | 14.63 | 1.83 |

### Stall shares, old → new

| stall reason | U old | U new | C old | C new | S old | S new | D old | D new |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MIO throttle | 42.34 | **0.33** | 31.45 | **0.25** | 1.89 | **0.16** | 14.60 | **0.00** |
| Barrier | 24.62 | **0.00** | 18.37 | **0.00** | 25.16 | **0.00** | 13.79 | **0.00** |
| Math pipe throttle | 3.66 | **21.78** | 3.24 | **11.14** | 7.01 | **15.99** | 0.81 | 0.70 |
| Wait | 8.13 | **20.63** | 12.73 | **30.63** | 16.26 | 19.39 | 23.33 | **41.55** |
| Not selected | 10.70 | **26.07** | 7.68 | 16.46 | 11.23 | 21.81 | 1.83 | 0.70 |
| Long scoreboard | 4.10 | 5.14 | 7.23 | 7.12 | 8.19 | 3.47 | 12.47 | 12.50 |
| No instruction | 3.81 | 5.53 | 5.59 | 4.75 | 15.20 | 19.27 | 10.43 | 7.61 |

### Uniform instruction mix: old → new

| family | old | new | change |
|---|---:|---:|---:|
| FFMA | 295.9M | 333.3M | +37.4M |
| LDS | 128.5M | 64.0M | **−64.5M** |
| FMUL | 69.0M | 41.8M | −27.2M |
| ISETP | 60.4M | 100.9M | **+40.5M** |
| IMAD | 43.0M | 133.8M | **+90.8M** |
| STS | 6.4M | 12.4M | +6.0M |
| BSYNC | 9.9M | 18.5M | +8.6M |
| BRA | 20.3M | 23.0M | +2.7M |
| **net** | **734.5M** | **854.6M** | **+120.2M** |

### Source groups (new capture)

| group | U inst % | C inst % | S inst % | D inst % |
|---|---:|---:|---:|---:|
| support index / modulo | **36.71** | **34.63** | **55.76** | **32.91** |
| Horner / FMA | 25.44 | 26.16 | 20.46 | 26.12 |
| other / kernel setup | 36.31 | 37.66 | 19.26 | 38.65 |
| bin iteration / offsets | 0.08 | 0.09 | 2.21 | 0.87 |
| point load | 0.54 | 0.55 | 0.87 | 0.55 |
| fold_rescale | 0.54 | 0.55 | 0.87 | 0.55 |
| ceilf | 0.36 | 0.37 | 0.58 | 0.36 |

### Uniform top source lines

| site | source | instructions | inst % | samples | sample % | top stall |
|---|---|---:|---:|---:|---:|---|
| `atomic_free_spread.cu:66` | `return covered ? ker[weight_index] : 0.0f;` | 150.4M | 17.60 | 59,570 | 25.56 | not selected |
| `atomic_free_spread.cu:202` | `acc.cell[...].x += xy * z.x` | 128.0M | 14.98 | 24,750 | 10.62 | wait |
| `atomic_free_spread.cu:203` | `acc.cell[...].y += xy * z.y` | 128.0M | 14.98 | 31,240 | 13.40 | not selected |
| `atomic_free_spread.cu:189` | `group.ky[j][cell_group.y + ly]` | 88.0M | 10.30 | 26,904 | 11.54 | wait |
| `ker_horner_allw_loop.inc:47` | Horner loop | 77.3M | 9.05 | 16,316 | 7.00 | not selected |
| `atomic_free_spread.cu:194` | `z[lz] = group.kz[j][...]` | 48.0M | 5.62 | 9,562 | 4.10 | wait |
| `atomic_free_spread.cu:54` | `offset += (offset < 0) ? grid_size : 0` | 41.7M | 4.88 | 7,283 | 3.12 | selected |
| `atomic_free_spread.cu:55` | `offset -= (offset >= grid_size) ? grid_size : 0` | 41.7M | 4.88 | 9,620 | 4.13 | not selected |
| `atomic_free_spread.cu:56` | `return offset < WIDTH ? offset : -1` | 25.3M | 2.96 | 8,493 | 3.64 | branch resolving |

### SASS families (new capture)

| family | U inst % | C inst % | S inst % | D inst % |
|---|---:|---:|---:|---:|
| FFMA | 39.00 | 40.28 | 24.33 | 40.22 |
| IMAD | 15.66 | 14.95 | 19.36 | 14.75 |
| ISETP | 11.81 | 11.00 | 17.69 | 10.75 |
| LDS | 7.49 | 7.79 | 2.47 | 7.78 |
| FMUL | 4.89 | 5.05 | 3.07 | 5.04 |
| IADD3 | 3.07 | 3.13 | 4.28 | 3.34 |
| SEL | 2.93 | 2.95 | 5.45 | 3.24 |
| BRA | 2.69 | 2.75 | 3.30 | 2.81 |
| BSYNC | 2.17 | 1.87 | 3.18 | 1.68 |
| STS | 1.45 | 1.46 | 2.31 | 1.46 |

### Assessment

1. **The microarchitectural hypothesis is confirmed.** The `2×2×4` lane cell group
   cut uniform LDS instructions by 50%, collapsed MIO throttle from 42.3% to
   0.33%, and removed barrier stalls entirely. Sparse improved 22% for the same
   reason. The LDS/barrier problem was real and this geometry fixes it.
2. **The prototype is still a net loss.** Uniform improves only 5.2%, clustered
   regresses 2.3×, and dense regresses 2.5×. Instructions increased 16.4% because
   the staging implementation more than consumed the LDS savings.
3. **The new bottleneck is staging instruction count.** One lane now stages all
   three axes for one source with only 16 active lanes. The old kernel used three
   staging lanes per source (one axis each) across three warps with 32 sources per
   group, so its warp-level staging cost per source was about 0.75 instructions;
   the new cost is about 1.5, a 2× increase. Support-index math rose from 169.2M
   to 313.8M instructions (+85%), IMAD from 43.0M to 133.8M (+3.1×), and ISETP from
   60.4M to 100.9M (+1.67×). The kernel is now FMA/math-pipe/issue-bound (FMA 76%,
   math-pipe throttle 21.8%), not LDS-bound.
4. **Clustered and dense fail for a structural reason.** One warp owns one region,
   so a hot region has one warp instead of the old four cooperating warps. There is
   no split mechanism to create more work items. Occupancy falls to 14.9%
   (clustered) and 4.3% (dense); wait stalls rise to 41% and 64% of stalls. No
   staging tweak fixes this; splits/active-region lists are required.
5. **Sparse confirms the same split.** Block barriers and MIO are gone, and sparse
   is 22% faster, but it is now issue/ALU-bound: support-index math is 55.8% of
   instructions. Coefficient staging and x-run batching are the next fixes.
6. **The gap to exp_a remains large.** exp_a is 1.69× faster on uniform, 5.44× on
   clustered, and 3.33× on sparse. exp_a stages 6 Horner coefficients plus support
   starts instead of 8 zero-filled weights per dimension, and it uses splits for
   clustered. Those are exactly the two missing pieces this profile exposes.

### Recommended next steps

1. **Do not adopt the prototype as-is.** It does not meet the uniform success
   criterion and regresses two of four cases.
2. **If iterating, fix the staging representation first.** Replace zero-filled
   per-cell weights with 6 Horner coefficients plus support placement, as exp_a
   does. This attacks the 36.7% support-index group and the 25.4% Horner/FMA group
   without changing the ownership geometry. Re-profile uniform and sparse.
3. **For clustered and dense, add splits/active-region lists.** One warp per
   region cannot recover hot-region parallelism. exp_a's `nsplit=56` clustered
   path is the reference.
4. **A cheaper intermediate experiment** is to use all 32 lanes for staging
   (`STAGING_GROUP_SIZE=32`, accepting 2-way `kz` bank conflicts and 9.2 KB shared
   per two-warp threadblock) or to pack four warp regions per threadblock. This may
   recover some uniform time but will not fix clustered/dense or the coefficient
   index math. Measure it only if coefficient staging is deferred.

## Next iteration: coefficient staging + active-region lists/splits

The user has approved both. They are not alternatives. Coefficient staging fixes
the uniform/sparse staging cost; active-region lists plus splits fix the
clustered/dense hot-region parallelism. Do not make either optional again.

### Phase 1 — coefficient staging (uniform and sparse)

Port exp_a's proven representation rather than inventing a new one:

- Use a staged-source record with three support starts, six x coefficients, six y
  coefficients, and six z coefficients, plus the zero padding slots needed for
  contiguous lane-group loads. Follow exp_a's `VALUES_PER_POINT = 30` layout.
- Port `place_support` / `place_support_nowrap`, `prepare_two_coefficients`, and
  `scan_one_staged_point` from `exp_a_cartesian.cu`, adapted to the current
  `WarpRegion`, `LaneCellGroup`, and `CellAccumulator`.
- Stage with nine lanes per source, three sources per warp iteration,
  `MAX_POINTS_PER_BATCH = 13`; keep one warp per warp region and `__syncwarp`.
- Zero the padded slots once per warp-private buffer and never overwrite them.
- Remove the zero-filled `StagingGroup`, `cell_weight`-based staging, and the
  per-cell `coefficient_index` path from the spread kernel.
- Keep the bench API and `launch_spread` signature unchanged.

Validation: full correctness matrix, then profile uniform and sparse. Expected:
remove most of the 36.7% support-index group and reduce the staging Horner cost.
If uniform still does not improve materially, the one-warp-per-region structure
itself is still suspect and must be revisited.

**Phase 1 result:** implemented and correct (9/9 matrix, 25/25 sweep, 144 edge
runs). Bench medians:

| case | Phase 0 | Phase 1 | old baseline | exp_a |
|---|---:|---:|---:|---:|
| uniform | 18.451 ms | **17.004 ms** | 19.471 ms | 10.912 ms |
| clustered | 61.145 ms | **62.939 ms** | 26.782 ms | 11.232 ms |
| sparse | 4.635 ms | **2.183 ms** | 5.968 ms | 1.393 ms |
| dense | 20.676 ms | **27.961 ms** | 8.326 ms | n/a |

Uniform instructions fell 854.6M → 755.1M; sparse fell 258.9M → 109.2M.
Coefficient staging clearly helps uniform/sparse, but support-index math is still
33.7% of uniform instructions because `place_support` moved into the scan, and
clustered/dense are unchanged-to-worse. Phase 2 is required, not optional.

### Phase 2 — active-region list plus splits (clustered and dense)

Port exp_a's mechanism, not just a list:

- Detect active warp regions from the candidate-bin source counts. A region is
  active if any candidate bin contains at least one source; this is conservative
  and matches exp_a's active-warp/tile test.
- Compact active regions into a deterministic list.
- Choose `nsplit` from the active count using exp_a's rule: target about 2048 work
  items when the active count is below 512, otherwise 512; cap at 32 (or 56 if
  later justified).
- Work item = (active warp region, split). Each split scans a disjoint subset of
  the source ranges at staging-group granularity: start at `begin + split * B`,
  advance by `nsplit * B`.
- When `nsplit > 1`, allocate `nsplit * cell_count` partial grids. Each work item
  writes only its warp region's cells into its split slice. Reduce the splits for
  active regions; zero the output for inactive regions (memset or a separate
  zero-fill) so every cell is written.
- Keep the one-warp-per-region scan. The split supplies the missing hot-region
  parallelism; the active list keeps the work-item count and reduction small.

Concrete implementation shape for this codebase:

1. **Region list.** Compute `regions_x/y/z = ceil(grid.n*/8)` and
   `region_count = regions_x*regions_y*regions_z`. Add a device helper to decode
   a flat region ID back to `WarpRegion`.
2. **Active detection.** One thread per region enumerates its candidate bins and
   sums `bin_offsets[bin+1]-bin_offsets[bin]`. If the sum is nonzero, mark the
   region active. This is conservative and matches exp_a's active-warp test.
3. **Compaction.** A single-thread kernel walks regions in order, writes active
   region IDs to `active_regions[]`, and stores `active_count`. Deterministic,
   no atomics.
4. **Work items.** `work_id = blockIdx.x*WARPS_PER_THREADBLOCK + warp_id`;
   `region_slot = work_id / split_count`, `split_index = work_id % split_count`.
   Decode `region_id = active_regions[region_slot]`. Warps past
   `active_count*split_count` return after the kernel-start coefficient-table
   barrier.
5. **Split scan.** Factor the current per-region body into a
   `spread_region(region, split_index, split_count, region_output, ...)` device
   function. The batch loop becomes
   `for (base = begin + split_index*B; base < end; base += split_count*B)`.
   A split writes every cell of its warp region, so each partial slice is fully
   defined for active regions.
6. **Full-grid fast path.** If `split_count == 1` and
   `active_count == region_count`, launch the unified 1D work-item `spread` with
   `active_regions = nullptr`, `region_id = work_id`, and write directly to
   output. No memset, no active list. Keeping one kernel name also keeps the
   existing profiler's `regex:^spread$` filter working.
7. **Active-list single-split path.** If `split_count == 1` and
   `active_count < region_count`, `cudaMemsetAsync` the output to zero, then
   launch the unified `spread` with `split_count=1` writing directly to output.
   This skips empty regions while keeping every cell written.
8. **Split path.** If `split_count > 1`, allocate
   `split_count * cell_count` partial grids. Launch the active-list kernel with
   the split count; each work item writes its region's cells into its split
   slice. Then `cudaMemsetAsync` the output to zero and launch a reduction kernel
   that sums the splits for active regions only. Inactive regions remain zero.
9. **Split heuristic.** `split_target = active_count < 512 ? 2048 : 512`;
   `nsplit = 1; while (nsplit < 32 && active_count*nsplit < split_target)
   nsplit <<= 1;`. If `active_count == 0`, memset output and return.
10. **Temporaries.** Allocate the active flags/list/count and partial grids with
    `cudaMalloc` inside `launch_spread`; free them before returning. The benchmark
    timing will include the allocation, the host copy of `active_count`, the
    memset, and the reduction. Report those costs instead of hiding them.

Validation: full correctness matrix, then profile clustered and dense and the
full matrix. Expected: occupancy and wait stalls recover toward the old kernel;
clustered should move from 61 ms toward exp_a's 11.2 ms, not stay at 61 ms. Watch
the host sync from the `active_count` copy, the partial-grid allocation, and the
reduction cost; if they dominate, the next step is a persistent workspace.

**Phase 2 result:** implemented and correct. 9/9 matrix, 25/25 sweep, split path
exercised by clustered (64/512 active, split 32) and dense (8/512, split 32);
forced split counts 1,2,4,8,16,32 and clipped/edge grids pass. `spread`: 96
registers, 3.3 KB static shared, no spills. Profile capture:
`build/gpu/traces/af_profile_20261110_175243/`.

| case | old baseline | Phase 0 | Phase 1 | Phase 2 | exp_a | Phase 2 / exp_a |
|---|---:|---:|---:|---:|---:|---:|
| uniform | 19.471 ms | 18.451 ms | 17.004 ms | **16.699 ms** | 10.912 ms | 1.53× |
| clustered | 26.782 ms | 61.145 ms | 62.939 ms | **16.433 ms** | 11.232 ms | 1.46× |
| sparse | 5.968 ms | 4.635 ms | 2.183 ms | **2.239 ms** | 1.393 ms | 1.61× |
| dense | 8.326 ms | 20.676 ms | 27.961 ms | **1.938 ms** | n/a | n/a |

Phase 2 is the first configuration that beats the old baseline on all four cases:
uniform 1.17× faster, clustered 1.63× faster, sparse 2.67× faster, dense 4.30×
faster. It remains 1.46–1.61× behind exp_a on uniform/clustered/sparse.

Key Phase 2 metrics:

| metric | uniform | clustered | sparse | dense |
|---|---:|---:|---:|---:|
| warp instructions | 754.6M | 824.2M | 106.8M | 61.2M |
| compute % | 59.00 | 74.28 | 67.38 | 49.26 |
| L1/TEX % | 57.29 | 77.99 | 71.39 | 56.26 |
| IPC | 2.55 | 2.72 | 2.70 | 2.11 |
| achieved occupancy % | 47.73 | 54.99 | 57.18 | 38.48 |
| no eligible % | 34.29 | 31.08 | 31.46 | 39.97 |

Remaining bottlenecks and risks:

- Uniform: support-index math is still 33.7% of instructions and the
  `place_support` general path is hot. The non-wrapping fast path is not hit
  often enough with per-bin batches; x-run batching is the natural next fix.
- Clustered: L1/TEX 78% and compute 74%. The split path restored parallelism;
  the remaining gap is per-warp single-buffer staging and the split count.
- Sparse: Phase 2 is slightly slower than Phase 1 (2.18 → 2.24 ms) because the
  metadata path costs about 0.2 ms (≈9%) at 128³. The active-list single-split
  path skips empty regions, but detection/compaction/host-copy/memset are not
  free.
- Partial-grid allocation churn is the main robustness risk: dense showed a
  4.17 ms outlier against a 1.94 ms median; the worker observed 2.6–7.3 ms
  `cudaMalloc`/`cudaFree` outliers. A persistent workspace or a compact
  per-active-region partial layout is the next infrastructure step.
- The profile script filters `^spread$`, so `detect_active_regions`,
  `compact_active_regions`, and `sum_splits` are not in the NCU tables. Their cost
  is included in the bench medians and was measured separately by the worker.

Recommended next steps, in order:

1. Persistent workspace (or compact partial grids) to remove allocation
   outliers and cut the sparse metadata cost.
2. x-run batching to make staged batches spatially coherent, increase the
   non-wrapping support fast path, and reduce support-index instructions.
3. Double-buffered per-warp staging to hide the global load latency, as exp_a
   does.
4. Try a larger clustered split count (56) and/or block-shared staging for
   clustered, as exp_a does.
5. Add the metadata/reduction kernels to the profile script's kernel list so
   their cost is visible in future captures.

### Order and attribution

Implement Phase 1, validate, and profile uniform/sparse. Then implement Phase 2,
validate, and profile clustered/dense and the full matrix. Do not merge the two
into one unattributed profile. If Phase 1 does not move uniform, stop and revisit
the one-warp-per-region assumption before spending time on Phase 2.

## Code quality refactor (readability and modularity)

The combined kernel is now a ~1005-line translation unit. It is performance-driven
accretion: geometry, the 30-float coefficient record, staging, the scan dispatch,
active-region policy, reduction, and host allocation are interleaved. The goal of
this refactor is readability and modularity **without changing measured behavior**.

### Diagnosis

- `spread_region` takes 12 parameters and mixes region geometry, split policy,
  staging, scan-path selection, accumulation, and output selection.
- The 30-float `StagedSource` layout is a long block comment plus magic slot
  constants repeated throughout the scan.
- The scan has two near-duplicate paths (window reads vs periodic taps) and a
  `<bool, bool>` template dispatch in the hot loop.
- `launch_spread` is ~85 lines of allocation, host copy, memset, split heuristic,
  four path branches, and frees.
- Candidate-bin enumeration is duplicated between `SourceRangeCursor` and
  `detect_active_regions`.
- The coefficient table and its loader dominate the top of the .cu and obscure the
  algorithm.
- Names mix glossary terms and old terms: `MAX_POINTS_PER_BATCH`,
  `scan_one_staged_point`, `sum_splits`.
- Host and device concerns share one file; allocation/lifetime policy is inline.

### Target modules

| module | interface | hides |
|---|---|---|
| `atomic_free_spread.cuh` | `Point`, `Grid`, `WIDTH`, `launch_*` | public API only |
| `spread_geometry.cuh` | `WarpRegion`, `LaneCellGroup`, `CandidateBins`, `CandidateBinCursor`, `SourceRangeCursor`, `SupportPlacement` | periodic wrap, lane mapping, candidate-bin enumeration, support placement |
| `spread_staging.cuh` | `StagedSource`, `StagingLane`, `stage_raw_sources`, `prepare_coefficients`, `zero_staging_padding` | 30-float layout, 9-lane staging, padding invariant |
| `spread_scan.cuh` | `CellAccumulator`, `SupportPath`, `scan_staging_group`, `store_cell_group_global/local` | window vs periodic scan, 2x2x4 outer product, store predicate |
| `spread_active_regions.cuh` / `.cu` | `ActiveRegionPlan`, `SpreadWorkspace`, `plan_active_regions`, `reduce_splits` | detection/compaction, split policy, buffer lifetime |
| `spread_kernel_coefficients.cuh` | the 9x6 table | coefficient data |
| `atomic_free_spread.cu` | `spread` kernel + `launch_spread` | composition only |

Keep headers grouped by concept, not one function per header. The modules are
compile-time only (`__forceinline__`, POD structs); no virtual calls, heap, or
`std::` containers in device code.

### Specific refactors

1. **Support placement returns a struct.** Replace the three output references
   with `struct SupportPlacement { int cell_count, coefficient_offset,
   local_cell_offset; }`. The scan becomes three named placements instead of six
   local int declarations.
2. **`StagedSource` accessors.** Keep the raw `float values[30]` for layout, but
   expose `support_starts()`, `strength()`, and `coefficient_row(axis)`. The scan
   stops spelling `staged_point[Z_COEFFICIENT_SLOT + ...]`; the slot constants live
   only in `spread_staging.cuh`.
3. **`SupportPath` enum.** Name the three scan policies (`Nowrap`, `Periodic`,
   `Degenerate`). `scan_staging_group` dispatches on the enum; the caller no longer
   carries `<bool, bool>` template flags.
4. **`RegionWorkItem` struct.** Group `region`, `cell_group`, `split_index`,
   `split_count`, and the output target. `spread_region` takes the work item plus a
   small `RegionScanContext` (grid, sources, bin offsets, staging, coefficients)
   instead of 12 positional parameters.
5. **`CandidateBinCursor`.** Extract the bin-ID enumeration shared by
   `SourceRangeCursor` and `detect_active_regions`. `SourceRangeCursor` wraps it
   with `bin_offsets`; active detection reuses it directly. This removes the
   duplicated triple loop.
6. **`SpreadWorkspace` + `ActiveRegionPlan`.** Move allocation, caching, host
   copies, and split selection behind a host module. `launch_spread` becomes a
   short policy: plan active regions, choose a path, launch, reduce. The workspace
   also fixes the `cudaMalloc`/`cudaFree` outliers (dense max 4.17 ms vs 1.94 ms
   median).
7. **Compact partial grids.** Change the split path from
   `partial[split][global_cell]` to
   `partial[split][active_slot][local_cell]`. The spread kernel writes its warp
   region's local 512 cells; the reduction sums local cells and maps to the global
   index once. This removes the 67–536 MB full-grid partial allocation, the global
   index math in the split store, and most of the allocation risk.
8. **Glossary renames.** `MAX_POINTS_PER_BATCH` → `STAGING_GROUP_SIZE`;
   `scan_one_staged_point` → `scan_one_staged_source`; `sum_splits` →
   `reduce_splits`; `stage_raw_points` → `stage_raw_sources`;
   `scan_staged_batch` → `scan_staging_group`. Keep `work_id`, `region_id`, and
   `region_slot` but document the distinction.
9. **Move the coefficient table** to its own header and replace the inline loader
   with `load_kernel_coefficients(...)`.
10. **Host tests for the pure modules.** Lane-group partition, region-ID decode,
    candidate-bin coverage under wrap, `SupportPlacement` vs brute force, and
    `periodic_tap_index` vs brute force. These are the regression net for the
    geometry refactor.

### Suggested sequence

| step | change | validation |
|---|---|---|
| R0 | Add host tests for the current geometry/support functions. | tests pass; no kernel change |
| R1 | Extract `spread_geometry.cuh` (region, lane group, candidate bins, cursor, support placement). | 9/9 matrix, 25/25 sweep, spot bench |
| R2 | Extract `spread_staging.cuh` with `StagedSource` accessors. | ptxas registers/shared/spills unchanged; 9/9 + spot bench |
| R3 | Extract `spread_scan.cuh`, introduce `SupportPath` and `RegionWorkItem`. | 9/9 + spot bench |
| R4 | Extract `ActiveRegionPlan`/`SpreadWorkspace` and path-specific launch functions. | 9/9 + allocation-outlier check + spot bench |
| R5 | Compact partial grids. | 9/9, forced split counts, sparse/clustered/dense bench |
| R6 | Glossary renames, comments, README/profile update. | full matrix + profile |

Do not combine R1–R4 with performance changes. R5 is the one behavior-preserving
performance/robustness change and should be profiled on its own. After each step,
check ptxas registers/spills and run the bench; the numbers should stay within
run-to-run noise.

### Non-goals

- No new features, split heuristics, or scan math during the refactor.
- No abstraction that prevents inlining or changes register allocation.
- No one-function headers or a header per helper; group by concept.
- Do not hide the performance tricks; move their comments next to the code they
  explain, and keep a short module-level contract at the top.

### Open questions

1. Is the refactor strictly behavior-preserving first, or may R5 (compact partial
   grids) land in the same series?
2. How many files are acceptable: five headers + one .cu, or five headers + two
   .cu (`spread_kernel.cu`, `spread_launch.cu`)? Splitting .cu files requires
   updating `run_atomic_free_spread.sh` and the profiler's build command.
3. Are host tests required, or is the existing bench sufficient?
4. Is the F0 prototype in scope for the `CandidateBins`/`candidate_bins` move? It
   currently includes the shared header and uses `detail::CandidateBins`.
5. Do we rename to glossary terms now, or keep the current names to minimize churn?

## Milestones and next actions

- [Done] Step 0: design and implement the warp-owned kernel.
- [Done] Step 1: profile it; the LDS/MIO/barrier hypothesis is confirmed, but the
  prototype is a net loss because of staging cost and hot-region parallelism.
- [Done] Step 2: Phase 1 coefficient staging. Validated 9/9 + 25/25 + 144 edge
  runs; profiled at `build/gpu/traces/af_profile_20261110_173855/`. Uniform
  17.00 ms, sparse 2.18 ms, clustered 62.94 ms, dense 27.96 ms.
- [Done] Step 3: Phase 2 active-region list plus splits. Validated 9/9 + 25/25
  and forced split counts; profiled at
  `build/gpu/traces/af_profile_20261110_175243/`. Uniform 16.70 ms, clustered
  16.43 ms, sparse 2.24 ms, dense 1.94 ms.
- [Done] Step 4: the combined kernel is accepted as the new best atomic-free
  baseline; the one-warp-per-region structure no longer needs to be revisited.
- [In progress] Step 5: readability and modularity refactor (R0–R6). Performance
  and behavior are held constant except for the deliberate compact-partial-grid
  change in R5.
- [Open] Step 6: after the refactor, return to performance: x-run batching,
  double-buffered staging, or a larger clustered split count.
- [Open] Next action: agree on the refactor scope: strictly behavior-preserving
  first, or include R5 in the same series; one .cu or two; host tests; F0 scope.
