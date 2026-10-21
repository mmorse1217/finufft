# Destination-Owned NUFFT Spreading with Morton-Sorted Spatial Binning

## Status

Design sketch for a CUDA prototype intended to test whether destination-cell ownership plus Morton-ordered point binning can outperform the current atomic-based cuFINUFFT spreading paths.

The initial target is:

- 3D NUFFT type 1
- FP32 first, then FP64
- fixed kernel width for the prototype
- CUDA implementation
- benchmarked directly against current cuFINUFFT `gpu_method=3`

The design deliberately avoids materializing the full point-to-grid interaction graph.

---

## 1. Motivation

In a standard point-driven type-1 NUFFT spread, each nonuniform point contributes to approximately

\[
n_s^d
\]

grid cells, where:

- \(N\) is the number of nonuniform points,
- \(n_s\) is the spreading-kernel support width,
- \(d\) is the dimension.

For 3D and \(n_s=6\), each point contributes to up to

\[
6^3 = 216
\]

grid cells.

A point-driven GPU implementation therefore naturally becomes a scatter:

```text
point j
  ├─> grid[k0] += contribution
  ├─> grid[k1] += contribution
  ├─> ...
  └─> grid[k215] += contribution
```

Many points can update the same grid cell concurrently, so atomics or a secondary reduction stage are required.

cuFINUFFT reduces this cost using spatial binning and shared-memory accumulation. Its output-driven method removes shared-memory atomics during local accumulation, but overlapping local subgrids still require a global atomic merge.

The design proposed here inverts ownership:

> Each destination grid cell has exactly one writer.

The spreading operation becomes a gather/reduction performed inside a destination-owned CUDA workgroup. Points are spatially sorted first so the destination workgroup can efficiently find and stream only the source points whose compact support can reach its output tile.

The key trade is:

```text
contended global atomic updates
            ↓
coherent point reads + local arithmetic + ordinary stores
```

The number of mathematical point-grid interactions remains \(O(N n_s^d)\), but those interactions are not materialized in memory.

---

## 2. High-Level Pipeline

```text
nonuniform points
    │
    │ compute integer spatial bin coordinates
    ▼
Morton code per point
    │
    ▼
GPU radix sort by Morton code
    │
    │ histogram/scan/scatter internally
    ▼
Morton-sorted point array / permutation
    │
    ▼
run-length encode occupied Morton bins
    │
    │ prefix scans / boundary compaction
    ▼
compact bin-run table
    │
    ▼
destination-owned spreading kernel
    │
    │ one workgroup owns one output grid tile
    │
    │ stream all source bins intersecting tile + kernel halo
    │
    │ stage points cooperatively
    │
    │ each thread accumulates owned cells in registers
    ▼
single non-atomic write per grid cell
    │
    ▼
oversampled NUFFT grid
```

No \(N n_s^d\) contribution array is created.

---

## 3. Core Ownership Rule

Partition the oversampled uniform grid into non-overlapping destination tiles.

For example, in 3D:

```text
grid
┌────────┬────────┬────────┐
│ tile 0 │ tile 1 │ tile 2 │
├────────┼────────┼────────┤
│ tile 3 │ tile 4 │ tile 5 │
├────────┼────────┼────────┤
│ tile 6 │ tile 7 │ tile 8 │
└────────┴────────┴────────┘
```

Each destination tile is assigned to exactly one CUDA workgroup.

Within a workgroup, individual grid cells are assigned to exactly one thread or lane.

Therefore:

```text
destination cell -> exactly one thread -> exactly one final store
```

No other workgroup writes that cell.

This removes the global accumulation race by construction.

---

## 4. Spatial Binning

### 4.1 Integer source bins

Each nonuniform point \(x_j\) is mapped to the oversampled grid coordinate system and assigned to a regular Cartesian source bin:

\[
b_x = \left\lfloor \frac{i_x}{B_x} \right\rfloor,\qquad
b_y = \left\lfloor \frac{i_y}{B_y} \right\rfloor,\qquad
b_z = \left\lfloor \frac{i_z}{B_z} \right\rfloor.
\]

The source-bin size \(B_x,B_y,B_z\) is a tuning parameter.

It should normally be comparable to the spreading support or destination tile dimensions.

---

## 5. Morton Coding

Instead of row-major linearization of \((b_x,b_y,b_z)\), construct a Morton/Z-order code:

\[
M_j = \operatorname{Morton}(b_x,b_y,b_z).
\]

For 3D, this interleaves bits from the three integer coordinates:

```text
bx: x2 x1 x0
by: y2 y1 y0
bz: z2 z1 z0

Morton:
z2 y2 x2 z1 y1 x1 z0 y0 x0
```

The exact bit order is arbitrary as long as it is consistent.

Morton ordering provides:

- points in the same source bin share the same key,
- spatially nearby bins tend to remain nearby in sorted order,
- coarse spatial hierarchy is encoded in the high bits,
- occupied bins can be represented sparsely,
- no dense spatial hash is required.

The Morton code is only an indexing key. Kernel values still use the true point coordinates.

---

## 6. Sorting Without Global Bin-Counter Contention

The point array is sorted by Morton code.

Conceptually:

```text
(point_0, M_0)
(point_1, M_1)
...
        ↓ radix sort
(M=12, point_7)
(M=12, point_19)
(M=12, point_21)
(M=13, point_3)
...
```

A GPU radix sort can be implemented using repeated:

```text
local histogram
    ↓
prefix scan
    ↓
global offsets
    ↓
scatter
```

or delegated to CUB/CCCL for the CUDA prototype.

The important property is that this avoids a single global atomic counter per spatial bin as the primary grouping primitive.

The trade is additional streaming memory traffic in exchange for less contention and a spatially coherent final ordering.

---

## 7. Occupied-Bin Run Construction

After sorting, equal Morton keys are contiguous.

For example:

```text
keys:
12 12 12 12 13 13 19 19 19 27 ...
```

Detect boundaries:

```cpp
boundary[i] = (i == 0) || (key[i] != key[i - 1]);
```

Then use a prefix scan over `boundary[]` to assign a run index.

Compact run metadata:

```cpp
struct BinRun {
    uint64_t morton;
    uint32_t begin;
    uint32_t count;
};
```

Example:

```text
run 0: Morton 12 -> points [0,4)
run 1: Morton 13 -> points [4,6)
run 2: Morton 19 -> points [6,9)
run 3: Morton 27 -> points [9,...)
```

This gives a sparse active-bin representation.

Memory is:

\[
O(N + N_{\text{occupied bins}})
\]

rather than \(O(N_{\text{all bins}})\) or \(O(N n_s^d)\).

---

## 8. Finding Source Bins for a Destination Tile

The spreading kernel has compact support.

For a destination tile \(T\), its output cells occupy a known Cartesian region:

\[
[x_0,x_1]\times[y_0,y_1]\times[z_0,z_1].
\]

If the kernel has support radius \(r\), then only source points inside the expanded region

\[
[x_0-r,x_1+r]
\times
[y_0-r,y_1+r]
\times
[z_0-r,z_1+r]
\]

can affect that tile.

Therefore the destination workgroup can compute the finite set of source bins that can contribute.

No per-interaction index list is required.

The workgroup performs:

```text
destination tile
      ↓
expand by kernel halo
      ↓
enumerate intersecting source-bin coordinates
      ↓
convert coordinates to Morton keys
      ↓
look up occupied-bin runs
```

Run lookup options include:

1. dense table from Morton/bin ID to run,
2. binary search in the sorted occupied-run array,
3. hash table over occupied bins,
4. small auxiliary dense lookup when the bin grid is reasonably sized.

For the prototype, a dense `bin -> run` table is likely simplest if memory permits.

---

## 9. Destination-Owned Spread Kernel

Each workgroup owns one destination tile.

Each thread owns one or more grid cells within that tile.

Conceptually:

```cpp
__global__ void spread_destination_owned(
    const Point* sorted_points,
    const BinRun* runs,
    Complex* grid,
    ...)
{
    DestinationTile tile = tile_for_block(blockIdx.x);

    Cell cells[CELLS_PER_THREAD] =
        cells_owned_by_this_thread(tile, threadIdx.x);

    Complex accum[CELLS_PER_THREAD] = {0};

    for (SourceBin b : source_bins_overlapping(tile)) {

        BinRun run = lookup_run(b);

        for (int batch = run.begin;
             batch < run.end;
             batch += POINT_BATCH) {

            load_points_to_shared(sorted_points, batch);

            __syncthreads();

            for (Point p : shared_point_batch) {

                // True coordinates determine kernel values.
                Kernel1D kx, ky, kz;
                evaluate_separable_kernel(p, kx, ky, kz);

                for (owned cell g : cells) {
                    if (g lies in support(p)) {
                        accum[g] +=
                            p.strength *
                            kx[g.x] *
                            ky[g.y] *
                            kz[g.z];
                    }
                }
            }

            __syncthreads();
        }
    }

    for (owned cell g : cells)
        grid[g] = accum[g];
}
```

The final store is non-atomic because ownership is unique.

---

## 10. Point Staging and Broadcast

A point should not be independently loaded by every destination-cell thread.

The workgroup cooperatively loads a batch:

```text
global memory
     │
     │ cooperative coalesced load
     ▼
shared memory / registers
     │
     ├─> thread owning cell 0
     ├─> thread owning cell 1
     ├─> thread owning cell 2
     └─> ...
```

Typical staged point data:

```cpp
struct Point {
    float x, y, z;
    Complex strength;
};
```

or SoA form:

```text
x[]
y[]
z[]
real[]
imag[]
```

SoA may give better global-memory behavior.

A useful pipeline is:

```text
batch k     -> compute
batch k+1   -> prefetch
```

using double-buffered shared memory or `cp.async` on architectures where it helps.

This is where latency hiding can be introduced without changing the algorithm.

---

## 11. Separable Kernel Evaluation

The spreading kernel is separable:

\[
K(x,y,z)=K_x(x)K_y(y)K_z(z).
\]

Therefore a point does not require \(n_s^3\) independent expensive kernel evaluations.

For each staged point compute:

\[
K_x[0:n_s],\qquad
K_y[0:n_s],\qquad
K_z[0:n_s].
\]

Then each 3D contribution is formed from products:

\[
K_{ijk}=K_x[i]K_y[j]K_z[k].
\]

This is particularly suitable for destination ownership because a staged point is reused by many cell-owning lanes.

Potential placements:

- per-point values in registers,
- shared memory for reuse by several warps,
- recompute lightweight factors if register pressure is excessive.

---

## 12. Interaction Count vs Intermediate Storage

The algorithm does **not** reduce the number of mathematical interactions.

For type 1:

\[
\text{interactions} \sim N n_s^d.
\]

For \(n_s=6,d=3\):

\[
216N
\]

point-cell contributions still exist mathematically.

The algorithm changes where those interactions are represented.

### Explicit interaction-list approach

```text
(point, destination cell, weight)
```

requires approximately:

\[
O(N n_s^d)
\]

temporary records.

### Destination-owned approach

Store only:

```text
points                  O(N)
Morton keys             O(N)
sorted permutation      O(N)
occupied bin runs       O(B_occ)
```

and regenerate interactions inside the kernel.

Thus:

\[
\text{intermediate memory}
=
O(N + B_{\text{occ}})
\]

while arithmetic remains \(O(N n_s^d)\).

---

## 13. Point Re-Reads Across Destination Tiles

A point near a tile boundary may influence more than one destination tile.

That means the point can be read by several workgroups.

If the destination tile width is \(B\) and kernel support width is \(n_s\), then the approximate average number of tiles touched per point in \(d\) dimensions is

\[
\left(1+\frac{n_s-1}{B}\right)^d
\]

for a uniform point distribution, ignoring wraparound details.

Example:

\[
n_s=6,\quad B=16,\quad d=3
\]

gives approximately

\[
\left(1+\frac{5}{16}\right)^3
\approx 2.26.
\]

So the algorithm may reread each point roughly 2–3 times on average while still avoiding 216 global atomic updates per point.

This is one of the central performance tradeoffs.

---

## 14. Atomics Removed

### Current point-driven style

A naïve type-1 spread performs roughly:

\[
N n_s^d
\]

global complex atomic updates.

### cuFINUFFT shared/output-driven style

cuFINUFFT reduces the number and scope of atomics using local accumulation, but overlapping subgrids still require a global merge.

### Proposed design

The target steady-state spreading path has:

```text
point sorting:
    no global per-bin atomic histogram required if radix sort is used

local accumulation:
    no atomic updates

global fine-grid write:
    no atomic updates
```

The major synchronization operations become:

- radix-sort scan/scatter phases,
- workgroup barriers during staged point processing,
- kernel launch boundaries between preprocessing stages.

The spread kernel itself has no inter-workgroup reduction.

---

## 15. Type-2 / Interpolation Symmetry

Type 2 naturally has unique ownership by output point:

\[
c_j
=
\sum_k
f_k K(x_j-k).
\]

Each nonuniform point produces exactly one final output value.

The complementary implementation is:

```text
Morton-sorted point runs
       ↓
one workgroup handles spatially local points
       ↓
load corresponding grid tile + halo
       ↓
stage grid values
       ↓
each thread owns one or more nonuniform points
       ↓
register reduction over support
       ↓
single write c[j]
```

Thus the same spatial structures can serve both directions:

### Type 1

```text
output grid cells own reduction
points are staged/broadcast
```

### Type 2

```text
output points own reduction
grid cells are staged/broadcast
```

Shared infrastructure can include:

- Morton coding,
- point permutation,
- occupied-bin run metadata,
- halo/bin enumeration,
- separable kernel evaluation,
- periodic wrapping.

---

## 16. Periodic Boundaries

FINUFFT's oversampled grid is periodic.

A destination tile near a grid boundary can receive contributions from points whose wrapped stencil crosses the opposite side.

The implementation must map source-bin neighborhoods and cell coordinates modulo the fine-grid dimensions.

Two possible strategies:

1. explicit modular arithmetic during neighborhood enumeration,
2. duplicate only boundary-bin descriptors, not point data.

The prototype should use explicit wrapping first because it is simpler and less error-prone.

---

## 17. Morton Order Is Not the Ownership Rule

Morton order is a data-locality and indexing device.

The correctness rule is destination ownership.

These are independent ideas.

A useful experimental decomposition is therefore:

### Experiment A

```text
existing Cartesian/bin ordering
+
destination-owned gather spread
```

### Experiment B

```text
Morton radix ordering
+
destination-owned gather spread
```

This separates the benefit of:

1. eliminating global atomic accumulation,
2. improving spatial ordering and preprocessing behavior.

The first effect may dominate.

---

## 18. Why Not Explicit Reduce-by-Key on All Contributions?

One alternative is:

```text
for every point:
    generate all n_s^d destination-cell contributions
        ↓
    sort by destination cell
        ↓
    reduce_by_key
        ↓
    write grid
```

This is correct and atomic-free, but it materializes approximately

\[
N n_s^d
\]

records.

For 3D \(n_s=6\):

\[
216N
\]

records are required.

At \(N=10^6\), this means 216 million records before accounting for weights or values.

The destination-owned gather avoids this expansion entirely.

The regular compact-support structure acts as an implicit sparse matrix.

---

## 19. Expected Performance Model

The proposed design roughly has:

\[
T_{\text{new}}
=
T_{\text{Morton sort}}
+
T_{\text{run construction}}
+
T_{\text{point rereads}}
+
T_{\text{kernel arithmetic}}
+
T_{\text{local reduction}}
+
T_{\text{ordinary stores}}.
\]

The cuFINUFFT method-3 baseline is approximately:

\[
T_{\text{m3}}
=
T_{\text{binning}}
+
T_{\text{kernel arithmetic}}
+
T_{\text{shared accumulation}}
+
T_{\text{global atomic merge}}.
\]

The proposed algorithm wins if the cost of:

- radix-sort traffic,
- repeated point reads across tile boundaries,
- support checks,
- any redundant kernel work,

is lower than the cost of:

- current binning contention,
- global atomic merge,
- associated cache-line serialization and synchronization.

---

## 20. Likely GPU Advantages

The new path intentionally converts synchronization-heavy work into GPU-friendly work:

```text
less:
    contended global atomics
    write/write conflicts
    global reduction synchronization

more:
    coherent read-only traffic
    streaming radix passes
    register accumulation
    shared-memory broadcast
    regular arithmetic
    single-writer global stores
```

Modern GPUs generally tolerate additional arithmetic and coherent reads better than severe contention on a small set of output locations.

This is especially interesting for FP64, where atomic-add support and throughput may be less favorable than ordinary FP64 arithmetic.

---

## 21. Main Risks

### 21.1 Sparse destination tiles

If many destination tiles contain no relevant points, launching one workgroup per tile can waste work.

Possible mitigation:

- construct an active destination-tile list from occupied source bins and kernel halo,
- prefix-scan/compact active tile flags,
- dispatch only active tiles.

### 21.2 Highly clustered points

A few destination tiles may receive extremely many points.

Possible mitigation:

- split overloaded destination tiles into multiple reduction partitions,
- write partial tile results and perform a small second-stage reduction,
- use adaptive Morton-prefix subdivisions.

### 21.3 Register pressure

Keeping many output-cell accumulators per thread can reduce occupancy.

Tuning dimensions:

- cells per thread,
- tile dimensions,
- workgroup size,
- point batch size,
- kernel-value caching strategy.

### 21.4 Shared-memory pressure

Large point batches or cached kernel values can reduce occupancy.

### 21.5 Redundant support tests

A destination-owned gather can inspect point/cell pairs that ultimately do not interact.

Bin size and tile geometry should be chosen so the candidate region is close to the true support region.

### 21.6 Sort cost

For small \(N\), radix sorting may dominate.

Possible policy:

```text
small N     -> existing method
large N     -> Morton/destination-owned method
```

---

## 22. Load-Balancing Extensions

The fixed one-workgroup-per-destination-tile design is the simplest starting point.

A more advanced version can use Morton hierarchy.

Because Morton prefixes correspond to spatial regions:

```text
coarse Morton prefix
    ├─ child region
    ├─ child region
    ├─ child region
    └─ child region
```

the implementation can recursively split high-occupancy regions.

This could give:

```text
light tile:
    one workgroup

heavy tile:
    split into subtiles or multiple point partitions

empty tile:
    no workgroup
```

The prototype should not implement this initially.

---

## 23. Prefix-Scan Uses

Prefix scans appear naturally in several preprocessing stages:

### Radix sorting

Each radix pass uses histogram counts followed by offsets.

### Run-length encoding

```text
key transition flags
        ↓
exclusive scan
        ↓
run IDs
```

### Active destination tiles

```text
tile_has_points[]
        ↓
exclusive scan
        ↓
compact active_tile_list[]
```

### Heavy-tile expansion

If a tile requires multiple work items:

```text
work_items_per_tile[]
        ↓
exclusive scan
        ↓
global work-item offsets
```

This keeps preprocessing mostly in bulk-parallel scan/scatter operations rather than globally contended increments.

---

## 24. Minimal CUDA Prototype

The first experiment should intentionally ignore most of FINUFFT.

### Scope

- 3D only
- type 1 only
- FP32 only
- one fixed kernel width, e.g. \(n_s=6\)
- one oversampling factor
- no FFT
- no public API integration
- CUB radix sort
- benchmark against cuFINUFFT spread method 3

### Prototype stages

```text
1. generate / import test points
2. compute Cartesian source-bin coordinates
3. compute Morton keys
4. CUB radix-sort (key, point-index)
5. construct bin runs
6. launch destination-owned spread
7. compare fine-grid result against cuFINUFFT
8. benchmark kernel and total preprocessing+spread time
```

### Correctness tests

Use:

- uniform random points,
- clustered points,
- repeated coordinates,
- boundary/wraparound points,
- very sparse regions,
- heavily occupied bins.

Measure:

\[
\frac{\|f_{\text{new}}-f_{\text{reference}}\|}
{\|f_{\text{reference}}\|}.
\]

Floating-point summation order differs from atomic spreading, so bitwise equality is not expected.

---

## 25. Benchmark Matrix

For the first meaningful performance evaluation:

### Point counts

```text
1e4
1e5
1e6
1e7
```

### Distributions

```text
uniform random
Gaussian cluster
multiple clusters
surface/shell
highly concentrated single region
```

### Kernel widths

```text
4
6
8
```

### Destination tile sizes

Example candidates:

```text
8x8x8
16x8x8
16x16x8
16x16x16
```

subject to register/shared-memory limits.

### Metrics

Measure:

- preprocessing time,
- spread-kernel time,
- total time,
- effective point throughput,
- DRAM bandwidth,
- L2 hit rate,
- achieved occupancy,
- registers/thread,
- shared-memory use,
- warp stall reasons,
- active destination tiles,
- average source bins per destination tile,
- average point reread factor,
- numerical error.

---

## 26. Success Criteria

The design is worth integrating further if it can satisfy most of:

1. Beat cuFINUFFT method 3 on large 3D type-1 workloads.
2. Remain competitive on uniform distributions.
3. Show a larger advantage on clustered distributions.
4. Avoid severe performance cliffs for sparse distributions.
5. Maintain acceptable temporary-memory use.
6. Extend naturally to FP64.
7. Preserve enough structure to reuse the spatial preprocessing for type 2.

A compelling result would be a consistent >20% spread-stage speedup with no major pathological regressions.

---

## 27. Recommended Experimental Order

```text
A. Cartesian bins + destination ownership
        ↓
B. Morton sort + destination ownership
        ↓
C. tune destination tile size
        ↓
D. double-buffer staged point loads
        ↓
E. kernel-value reuse / register tuning
        ↓
F. active-tile compaction
        ↓
G. heavy-tile splitting
        ↓
H. FP64
        ↓
I. type 2
        ↓
J. integrate into cuFINUFFT plan/execution path
```

This order isolates the source of each performance improvement.

---

## 28. Summary

The proposed algorithm combines two ideas:

1. **Morton-sorted spatial binning**
   - spatially coherent point ordering,
   - scan-based radix construction,
   - compact occupied-bin runs,
   - no need to build a full sparse interaction graph.

2. **Destination-owned spreading**
   - one writer per grid cell,
   - points streamed through destination workgroups,
   - register/shared-memory reduction,
   - direct non-atomic final stores.

The central observation is:

\[
\boxed{
\text{The stencil contains }N n_s^d\text{ interactions, but they do not need }
N n_s^d\text{ storage locations or atomic updates.}
}
\]

The known compact support and regular grid provide enough structure to regenerate those interactions inside a destination-owned kernel.

The resulting design trades global atomic contention for coherent reads, local arithmetic, register reduction, and ordinary stores. That is a plausible GPU performance win and is small enough to evaluate with a focused CUDA prototype before committing to a larger FINUFFT or Vulkan implementation.
