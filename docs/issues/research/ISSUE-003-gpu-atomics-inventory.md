# ISSUE-003 research: cuFINUFFT GPU atomic and bin-sort inventory

**Ticket:** [ISSUE-003](../done/ISSUE-003-gpu-atomics-inventory.md)
**Revision examined:** `ee272845` ("order py err_handler dict to match finufft_error.h", 2026-10-20);
project version 2.6.0 (`CMakeLists.txt:2`).
**Method:** source reading only. No build or profile was run for this ticket. All line numbers are
as of the revision above; paths are relative to the repo root.

This document inventories the four cuFINUFFT spreading paths (`gpu_method` 1, 2, 3, 4) and their
preprocessing. It is a fact inventory: it does not choose a design.

---

## 1. Where the four paths live, and when they run

Dispatch is by `opts.gpu_method` in three plan methods:

| Stage | Function | Location |
|---|---|---|
| preprocessing (once per `setpts`) | `cufinufft_plan_t<T>::indexSort()` | `src/cuda/spreadinterp.cpp:14-30` |
| spreading (per batch of transforms) | `cufinufft_plan_t<T>::spreadSorted()` | `src/cuda/spreadinterp.cpp:32-48` |
| interpolation | `cufinufft_plan_t<T>::interpSorted()` | `src/cuda/spreadinterp.cpp:50-62` (only methods 1 and 2) |

`setpts_type12()` allocates the per-point arrays, stores the coordinate pointers, then calls
`indexSort()`: `src/cuda/setpts.cu:65-76` (`allocate_nupts()` at `:68`, `indexSort()` at `:72`). For
type 3, `setpts()` allocates the outer (type-1) plan's subproblem state and calls
`setpts_type12()` at `src/cuda/setpts.cu:300-302`, and builds an inner type-2 plan at `:304-317`.
`execute_type1()` zeroes `fw` and then calls `spreadSorted()`: `src/cuda/execute.cu:135-138`;
type 3 does the same at `src/cuda/execute.cu:238-246`.

Consequences that matter for every cost below:

- **Preprocessing is paid once per `setpts`** and is reused by all `ntransf` transforms and by
  repeated `execute()` calls on the same point set.
- **Spreading is paid once per batch of transforms.** Every spread driver loops
  `for (t = 0; t < blksize; t++)` and launches one kernel per transform (e.g.
  `src/cuda/spread_nupts_driven_inst.cu:73-77`, `src/cuda/spread_subprob_inst.cu:107-110`,
  `src/cuda/spread_output_driven_inst.cu:120-124`, `src/cuda/spread_blockgather_inst.cu:292-295`).
- `fw` is zeroed by the execute path before spreading (`src/cuda/execute.cu:135`), and every spread
  path *adds into* it.

Per-method implementation units:

| `gpu_method` | Docs name | Preprocessing driver | Spread kernel | Dims |
|---|---|---|---|---|
| 1 | GM / GM-sort | `do_indexSort_nupts_driven` — `src/cuda/spread_nupts_driven_inst.cu:102-130` | `spread_nupts_driven` — `src/cuda/spread_nupts_driven_inst.cu:17-62` | 1,2,3 |
| 2 | SM | `do_indexSort_subprob_and_OD` — `src/cuda/spread_subprob_inst.cu:136-181` | `spread_subprob` — `src/cuda/spread_subprob_inst.cu:17-97` | 1,2,3 |
| 3 | OD | same as method 2 | `spread_output_driven` — `src/cuda/spread_output_driven_inst.cu:17-103` | 1,2,3 |
| 4 | 3-D block gather | `do_indexSort_blockgather_3d` — `src/cuda/spread_blockgather_inst.cu:323-443` | `spread_3d_block_gather` — `src/cuda/spread_blockgather_inst.cu:168-275` | 3 only (`src/cuda/makeplan.cu:151-155`, `src/cuda/spread_blockgather_inst.cu:171`) |

Relevant option defaults (`src/cuda/c_interface.cpp:158-170`): `gpu_method=0` (auto),
`gpu_sort=1`, `gpu_maxsubprobsize=1024`, all bin sizes `0` (heuristic), `gpu_np=0`,
`gpu_kerevalmeth=1`. Auto-method is 2 for types 1 and 3 and 1 otherwise (`src/cuda/makeplan.cu:306-320`).
`opts.gpu_np` is forced to 0 for every method except 3 (`src/cuda/makeplan.cu:270`), so the
`np` term of `shared_memory_required()` is zero for methods 1, 2, and 4.

---

## 2. Counting conventions used below

- `atomicAddComplexShared()` is **always two `atomicAdd_block` calls**, one per real/imag component
  (`include/cufinufft/utils.hpp:78-84`). No native complex shared-memory atomic is used.
- `atomicAddComplexGlobal()` (`include/cufinufft/utils.hpp:91-102`) is **one `atomicAdd(float2)` only
  for FP32 when `__CUDA_ARCH__ >= 900`** (`:94-96`, macro at `:56-65`). Otherwise it is **two scalar
  `atomicAdd` calls**. On the map's sm_75 Quadro T1000, therefore, every complex global atomic in
  these paths is two hardware atomics; on sm_90+ FP32 it is one 64-bit vector atomic.
- **Charge units:** "per point" = per nonuniform point, once per `indexSort()`; "per interaction" =
  per (point, grid cell) kernel tap, i.e. up to `ns^dim` per point; "per padded subgrid cell" = per
  cell of a block's shared tile during the local→global merge; "per obin cell" = per cell of a
  method-4 output bin.
- The exhaustive set of atomic sites in the CUDA sources is exactly the list in §9 (grep of
  `atomicAdd` over `src/cuda` and `include/cufinufft`; the only non-spread ones are the
  `atomicAdd_block`/`atomicAdd` helpers themselves).

---

## 3. Arrays that the preprocessing produces

Declared in `include/cufinufft/cufinufft_plan_t.hpp:357-374` (device copies in
`cufinufft_gpu_data`, `:565-580`); allocated per method in `src/cuda/makeplan.cu:127-177`
(`allocate_subprob_state`) and `:181-209` (`allocate_nupts`).

| Array | Meaning (from the header comments) | Size by method |
|---|---|---|
| `idxnupts` | point indices in bin-sorted order; the spread kernels iterate this | 1: `M`; 2/3: `M`; 4: not set in `allocate_nupts`, resized later to the **number of bin entries including ghost copies** (`src/cuda/spread_blockgather_inst.cu:399-403`) |
| `sortidx` | rank of the point **inside its bin** | 1: `M` if `gpu_sort`, else 0; 2/3: `M`; 4: `M` |
| `binsize` | points (entries) per bin | 1: `nbins` if `gpu_sort`, else 0; 2/3: `nbins`; 4: expanded bin count `Π numobins·(obin/bin+2)` |
| `binstartpts` | exclusive scan of `binsize` | same as `binsize`; method 4 has one extra element (`src/cuda/makeplan.cu:168`) |
| `numsubprob` | subproblems per bin (per outer bin for method 4) | 2/3: `nbins`; 4: `numobins` |
| `subprobstartpts` | exclusive scan of `numsubprob` (length +1) | 2/3: `nbins+1`; 4: `numobins+1` |
| `subprob_to_bin` | bin (outer bin for method 4) served by each subproblem | 2/3 and 4: `totalnumsubprob` |

`nbins = Π ceil(nf_i / binsize_i)` (`include/cufinufft/spreadinterp.hpp:138-149`), computed on
host+device by `compute_bin_layout` (`src/cuda/spreadinterp_common.cuh:48-64`). Note that `nbins`
uses `ceil`, so the last bin along an axis can be narrower than `gpu_binsize*`; methods 2 and 3 do
**not** require `binsize | nf`. Method 4 does require exact divisibility: `nf % obinsize == 0` and
`obinsize % binsize == 0` are validated and throw `FINUFFT_ERR_BINSIZE_NOTVALID`
(`src/cuda/spread_blockgather_inst.cu:332-359`).

Methods 2, 3, and 4 must size `subprob_to_bin` on the host, so `setpts` contains a device→host copy
and a full stream synchronisation: `src/cuda/spread_subprob_inst.cu:168-174` and
`src/cuda/spread_blockgather_inst.cu:432-438`.

---

## 4. Method 1 (nonuniform-point-driven; GM / GM-sort)

### 4.1 Preprocessing — `do_indexSort_nupts_driven`

`src/cuda/spread_nupts_driven_inst.cu:102-130`. With `gpu_sort=1`:

1. `cudaMemsetAsync(binsize, 0, nbins*sizeof(int))` — `:109-110`.
2. `calc_bin_size_noghost<<<ceil(M/1024),1024>>>` — `:111-114`, body in
   `include/cufinufft/spreadinterp.hpp:299-311`. For each point it computes a **row-major flat bin
   index** from the rescaled coordinates (`compute_bin_index`, `spreadinterp.hpp:215-229`; the index
   is clamped into `[0, nbins_i-1]` at `:223-224`), then
   `oldidx = atomicAdd(&bin_size[binidx], 1)` (`:308`) and stores `oldidx` into `sortidx[i]`
   (`:309`).
3. `thrust::exclusive_scan(binsize → binstartpts)` — `:116-117`.
4. `calc_inverse_of_global_sort_idx` — `:120-124`, body in `spreadinterp.hpp:313-325`. It
   **recomputes the same bin index** (`:321`) and writes
   `index[binstartpts[binidx] + sortidx[i]] = i` (`:322-323`). The write is **not** atomic: the pairs
   `(binstartpts[binidx], sortidx[i])` are unique by construction because `sortidx` is the rank
   returned by the histogram atomic. This pass is what turns the rank array into the grouped point
   permutation `idxnupts`.

With `gpu_sort=0`: `thrust::sequence(idxnupts)` only (`:126-128`); no histogram, no scan, no
atomics, and `binsize`/`binstartpts` were never allocated (`src/cuda/makeplan.cu:133-139`).

**Atomic cost:** exactly **one global `atomicAdd(int)` per NU point, once per `setpts`**
(`spreadinterp.hpp:308`), and only in the sorted variant. This is a per-point cost, not per
interaction.

### 4.2 Spreading — `spread_nupts_driven`

`src/cuda/spread_nupts_driven_inst.cu:17-62`. Each thread owns one point per grid stride; it reads
the permutation `idxnupts[i]` (`:27`), evaluates the 1-D kernel factors and start offsets
(`get_kerval_and_startpos_nuptsdriven`, `include/cufinufft/spreadinterp.hpp:178-196`), and scatters
`ns^dim` taps with `atomicAddComplexGlobal`:

- 1-D: `:33-35`, `ns` global atomics per point.
- 2-D: `:37-44`, `ns^2` global atomics per point.
- 3-D: `:46-58`, `ns^3` global atomics per point.

Periodic wrap is handled *inside* the tap loops (`ix = (ix + 1 >= nf) ? 0 : ix + 1`, etc.), so no
index arithmetic beyond the loop update. Launch geometry: 16 threads/block,
`ceil(M/16)` blocks, one launch per transform (`:69-77`).

**Atomic cost:** **one global complex atomic per point–cell interaction**, i.e. `ns^dim` per point
per transform. This is the only path whose spread cost is entirely per-interaction global atomics.

---

## 5. Methods 2 and 3 — identical preprocessing

`do_indexSort_subprob_and_OD` (`src/cuda/spread_subprob_inst.cu:136-181`) is used by both methods
and is unconditional (there is no `gpu_sort` switch for methods 2/3):

1. `cudaMemsetAsync(binsize)` — `:142-143`.
2. `calc_bin_size_noghost` histogram — `:144-147`. Same kernel and same per-point
   `atomicAdd(int)` as method 1 (`spreadinterp.hpp:299-311`, atomic at `:308`), same row-major
   `compute_bin_index` (`:215-229`).
3. `thrust::exclusive_scan(binsize → binstartpts)` — `:148-150`.
4. `calc_inverse_of_global_sort_idx` — `:151-155`, non-atomic scatter (`spreadinterp.hpp:313-325`).
5. `calc_subprob` — `:160-162`, body at `spreadinterp.hpp:333-340`:
   `numsubprob[i] = ceil(binsize[i] / maxsubprobsize)` (default `maxsubprobsize=1024`).
6. `thrust::inclusive_scan(numsubprob[0..nbins) → subprobstartpts[1..])` — `:164-166`, then
   `subprobstartpts[0] = 0` — `:169`.
7. Device→host copy of `subprobstartpts[nbins]` (= `totalnumsubprob`), `cudaStreamSynchronize`,
   resize of `subprob_to_bin` — `:170-174`.
8. `map_b_into_subprob` — `:176-178`, body at `spreadinterp.hpp:341-349`: for each bin, fill
   `subprob_to_bin[subprobstartpts[bin] + j] = bin` for `j < numsubprob[bin]`. No atomics; each
   subproblem slot has exactly one writer.

At spread time, a block resolves itself with `compute_subprob_block_info`
(`src/cuda/spreadinterp_common.cuh:28-43`): `bidx = subprob_to_bin[subprob]` (`:34`),
`binsubp_idx = subprob - subprobstartpts[bidx]` (`:35`),
`ptstart = binstartpts[bidx] + binsubp_idx*maxsubprobsize` (`:36-37`), and
`nupts = min(maxsubprobsize, binsize[bidx] - binsubp_idx*maxsubprobsize)` (`:38-40`). Thus a bin
with more than `maxsubprobsize` points is split into several blocks, each still writing the **whole**
bin tile (see §§6–7, 11). `gpu_maxsubprobsize` is documented as method-2-only
(`docs/c_gpu.rst:425`), but method 3 uses the same helper and the same subproblem partition.

**Atomic cost of the shared preprocessing:** **one global `atomicAdd(int)` per NU point** per
`setpts` (same kernel/line as method 1), plus O(nbins) scan/memset work and one host synchronisation
per `setpts`.

---

## 6. Method 2 — `spread_subprob` (SM)

`src/cuda/spread_subprob_inst.cu:17-97`.

1. One block per subproblem (`grid = totalnumsubprob`, `:108`), 256 threads (`:108`), dynamic
   shared memory = `shared_memory_required()` = padded tile only (`:103`), because `gpu_np=0` for
   this method.
2. Compute the block's bin, point range, and bin offset (`:28-33`), then zero the padded tile of
   volume `V_bin = Π (binsize_i + 2·ceil(ns/2))` (`:35-43`) and `__syncthreads()` (`:45`).
3. Point loop is **point-parallel**: `for (i = threadIdx.x; i < nupts; i += blockDim.x)`
   (`:47`). One thread computes all `ns` kernel values for its point
   (`get_kerval_and_local_start`, `include/cufinufft/spreadinterp.hpp:153-172`) and performs all
   `ns^dim` tap updates. Two threads can hit the same tile cell, so the accumulation uses
   `atomicAddComplexShared` (`:57` 1-D, `:67` 2-D, `:83` 3-D) = 2 shared atomics per tap.
4. `__syncthreads()` (`:90`), then `shared_mem_copy_helper` (`:93-96`) walks every padded cell and
   does `atomicAddComplexGlobal(fw + globidx, fwshared[n])` (`:95`). The helper
   (`include/cufinufft/spreadinterp.hpp:269-294`) maps each local cell to the global grid with
   `idx = idx0 + offset - ns/2` (`:284`) and a single periodic wrap (`:286`); it skips a cell when
   `idx >= nf + ns/2` (`:285`). So the merge covers **every cell of the padded tile, including the
   `ns/2`-wide collar and cells whose contribution is zero**.

**Atomic cost per transform:** `ns^dim` **shared** complex atomics per point (2 hardware atomics
each) **plus** `V_bin` **global** complex atomics per subproblem block. The global count per point is
`V_bin·ceil(occ/maxsubprobsize)/occ` for a bin holding `occ` points, i.e. between `V_bin/maxsubprobsize`
(dense bin) and `V_bin` (bin with a single point).

---

## 7. Method 3 — `spread_output_driven` (OD)

`src/cuda/spread_output_driven_inst.cu:17-103`.

1. Same block→bin/subproblem mapping and same launch grid as method 2
   (`compute_subprob_block_info` at `:27`; `grid = totalnumsubprob` at `:121`). Thread count is
   `min(256, max(ns^dim, gpu_np))` (`:114`, `:121`); for 3-D `ns=6`, `ns^dim=216`.
2. Shared memory layout (`:40-50`): `kerevals` (`np` × `dim` × `ns` values), `nupts_sm` (`np`
   complex strengths), `shift` (`np` × `dim` ints), then `local_subgrid` of volume
   `V_bin = Π (binsize_i + 2·ceil(ns/2))` (`:38`, `get_padded_subgrid_info` at
   `include/cufinufft/spreadinterp.hpp:231-242`). This matches `shared_memory_required()` exactly
   (`src/cuda/heuristics.cu:93-111`).
3. Zero `local_subgrid` (`:53-55`), `__syncthreads()` (`:57`).
4. For each batch of `np` points (`:59-60`): cooperatively stage strengths and per-point kernel
   values/offsets into shared (`:61-68`), `__syncthreads()` (`:69`), then **loop over the batch's
   points serially** (`:71`). For each point, all threads cooperate over the `ns^dim` taps
   (`:75-93`) and update `local_subgrid[idxout] += cnow*kervalue` (`:92`) with **no atomic**. The
   flat tap index maps to a unique tile index, so different threads do not collide *within one
   point*; different points are serialised, and a `__syncthreads()` at `:94` closes each point.
5. After all points, write the whole padded tile with `atomicAddComplexGlobal` (`:98-102`), using
   `output_index_from_flat_local_index` (`spreadinterp.hpp:246-262`), which wraps every cell once
   modulo `nf` (`:255-256`). There is **no `in_region` guard** here, unlike method 2.

**Atomic cost per transform:** **zero shared atomics**, one non-atomic shared read-modify-write per
interaction, one block barrier per point, plus `V_bin` **global** complex atomics per subproblem
block (same formula as method 2). This is the existing path that already eliminates the local
(shared-memory) atomics; `docs/impl_gpu.rst:76-86` states the same intent ("removes the need for
`AtomicAdd` on the local subgrid") and that the global atomic merge is "unchanged from SM".

---

## 8. Method 4 — `spread_3d_block_gather` (3-D block gather)

### 8.1 Geometry and preprocessing

Two bin sizes: fine bins `gpu_binsize*` (default 4) and "outer" bins `gpu_obinsize*` (default 8)
(`src/cuda/heuristics.cu:345-352`). Outer bins partition the fine grid; `m = obin/bin` real fine
bins per dimension per outer bin, plus **two ghost fine bins per dimension** (one on each side), so
`binsperobin_i = obin_i/bin_i + 2` (`src/cuda/spread_blockgather_inst.cu:186-191`,
`src/cuda/makeplan.cu:158-164`). A point is located into a real fine bin with the padded numbering
`bin = (bin/m)*binsperobin + (bin%m + 1)`, `m = binsperobin-2 = obin/bin`
(`src/cuda/spread_blockgather_inst.cu:132-137`), so real bins occupy local indices `1..m` of each
outer bin and ghosts are local `0` and `m+1`.

`do_indexSort_blockgather_3d` (`src/cuda/spread_blockgather_inst.cu:323-443`):

1. `cudaMemsetAsync(binsize, 0, numbins_expanded)` — `:374-375`.
2. `locate_nupts_to_bins_ghost` — `:377-379`, body `:120-143`: per-point padded bin remap
   (`:132-137`), row-major flat index over the expanded bin grid (`calc_global_index`, `:13-19`),
   **one global `atomicAdd(&bin_size[binidx], 1)` per point** (`:140`), store the within-bin rank in
   `sortidx` (`:141`). Only real bins receive points, so `Σ binsize(real) = M`.
3. `fill_ghost_bins` — `:388-390`, body `:45-75`: for each ghost bin, **plain copy** of the count
   from the corresponding real bin on the other side of the outer-bin boundary
   (`binsize[binidx] = binsize[idxtoupdate]`, `:71-72`). No atomic; a separate kernel, so it is
   ordered after the histogram on the same stream. After this, `Σ binsize = M + duplicates`.
4. `thrust::inclusive_scan(binsize → binstartpts[1..])` — `:392-397`, then D2H of the last element,
   sync, and `idxnupts.resize(total entries)` — `:399-403`.
5. `calc_inverse_of_global_sort_index_ghost` — `:405-407`, body `:145-166`: recomputes the padded
   bin index and writes `index[binstartpts[binidx] + sortidx[i]] = i` (no atomic; unique slots).
6. `ghost_bin_pts_index` — `:415-416`, body `:77-118`: for each ghost bin, **plain copy** of the
   source real bin's entries into the ghost bin's range, prefixing a 3-digit base-3 "box" code:
   `index[...] = M*(box0 + 3*box1 + 9*box2) + index[source...]` (`:113-114`). The box code is
   non-zero only for the periodic wrap at the grid boundary (`:97-108`); it is later decoded into a
   `±nf` coordinate shift (`spread_3d_block_gather`, `:217-229`). No atomics; disjoint output
   ranges.
7. Subproblem construction per **outer** bin: `calc_subprob_3d_v1` sums `binsize` over **all**
   expanded bins of the outer bin (including ghosts) and sets
   `numsubprob[obin] = ceil(entries/maxsubprobsize)` (`:21-34`, call `:421-425`); inclusive scan
   into `subprobstartpts[1..]`, D2H sync, `subprob_to_bin.resize`, then
   `map_b_into_subprob_3d_v1` (`:36-43`, call `:439-440`). No atomics.

**Duplication factor (derived from the ghost conditions at `:61-68` and `:97-108`):** each ghost
layer copies exactly one real fine bin, so a point is duplicated once per dimension in which its bin
is the first or last real bin of its outer bin. With the default `m = 8/4 = 2`, *both* real bins are
boundary bins, so every point is duplicated once per dimension → up to `2^3 = 8` entries per point in
3-D; with `m=4` half the points are duplicated per dimension → `1.5^3 ≈ 3.4` entries per point on
average. The interaction count does not grow with the replication (taps are clipped to the owning
outer bin, §8.2); what grows is `idxnupts` memory and per-entry setup work.
The ghost layer only supplies points from the *immediately adjacent* fine bin, which is sufficient
only while `bin >= ceil(ns/2)-1`; no code validates this together with `ns`.

### 8.2 Spreading

`src/cuda/spread_blockgather_inst.cu:168-275`.

1. Block ↔ subproblem; `obidx = subprob_to_bin[subpidx]` (`:196`); the block's point range is a
   contiguous slice of the outer bin's **expanded** bin range:
   `ptstart = binstartpts[bidx] + obinsubp_idx*maxsubprobsize`, where `bidx = obidx*binsperobin_tot`
   (`:197-203`). Block offset in the fine grid from `compute_offset` (`:205`).
2. Shared buffer is **exactly the outer bin**: `N = Π obin_i` complex, no padding (`:207`,
   `:211`); zero and `__syncthreads()` (`:211-213`).
3. Point loop is point-parallel (`:215`). Each entry decodes the box code (`:217-225`), rescales the
   coordinate and adds `box·nf` for periodic wrap (`:227-229`), evaluates the kernel
   (`:236-240`, start/end from `interval`, `include/cufinufft/utils.hpp:51-55`), and **clips the tap
   range to the outer bin**: `startnew = max(start,0)`, `endnew = min(end, obin-1)` (`:243-244`).
   It then does `atomicAddComplexShared` per tap (`:249-259`, atomic at `:255`) = 2 shared atomics
   per interaction.
4. `__syncthreads()` (`:261`), then write **exactly the outer bin's cells** with
   `atomicAddComplexGlobal` (`:264-273`, atomic at `:273`); no halo, no modulo.

Launch: `grid = totalnumsubprob`, 64 threads (`:293`), dynamic shared =
`Π obin_i · sizeof(complex)` (`:282-283`) with a **hard-coded 48 KB cap**
(`if (sharedplanorysize > 49152) throw FINUFFT_ERR_INSUFFICIENT_SHMEM`, `:284-288`), and the
`cufinufft_set_shared_memory` opt-in call is commented out (`:291`).

**Atomic cost per transform:** `ns^3` **shared** complex atomics per point entry (clipped at tile
edges; summed over replicas this is still ≈ one per mathematical interaction, because taps outside
the owning outer bin are skipped), **plus** `Π obin_i` **global** complex atomics per subproblem
block.

---

## 9. Consolidated atomic inventory

| # | Site (file:line) | Scope | Intrinsic | Count | Charged to | Phase |
|---|---|---|---|---|---|---|
| 1 | `include/cufinufft/spreadinterp.hpp:308` (`calc_bin_size_noghost`) | global | `atomicAdd(int)` | 1 | per NU point | preprocessing; methods 1 (if `gpu_sort`), 2, 3 |
| 2 | `src/cuda/spread_blockgather_inst.cu:140` (`locate_nupts_to_bins_ghost`) | global | `atomicAdd(int)` | 1 | per NU point (real bin placement only) | preprocessing; method 4 |
| 3 | `src/cuda/spread_nupts_driven_inst.cu:35`, `:43`, `:56` | global | `atomicAddComplexGlobal` → 1 vector atomic (FP32, sm≥90) or 2 scalar atomics | `ns^dim` | per point–cell interaction | spread; method 1 |
| 4 | `src/cuda/spread_subprob_inst.cu:57`, `:67`, `:83` | shared | `atomicAddComplexShared` → 2 `atomicAdd_block` | `ns^dim` | per point–cell interaction | spread; method 2 |
| 5 | `src/cuda/spread_subprob_inst.cu:95` (via `spreadinterp.hpp:269-294`) | global | `atomicAddComplexGlobal` | `V_bin` | per padded subgrid cell | spread merge; method 2 |
| 6 | `src/cuda/spread_output_driven_inst.cu:101` | global | `atomicAddComplexGlobal` | `V_bin` | per padded subgrid cell | spread merge; method 3 |
| 7 | `src/cuda/spread_blockgather_inst.cu:255` | shared | `atomicAddComplexShared` → 2 `atomicAdd_block` | up to `ns^3` per entry (tile-clipped) | per point-entry interaction | spread; method 4 |
| 8 | `src/cuda/spread_blockgather_inst.cu:273` | global | `atomicAddComplexGlobal` | `Π obin_i` | per obin cell | spread merge; method 4 |

with `V_bin = Π_i (binsize_i + 2·ceil(ns/2))`, `i = 1..dim` (from
`include/cufinufft/spreadinterp.hpp:231-242` and `src/cuda/heuristics.cu:102-111`).

No atomic in any path is charged **per bin**: the per-bin work (memset, prefix scan,
`calc_subprob`, `map_b_into_subprob`, `fill_ghost_bins`) is non-atomic and charged per bin, while
the histogram atomic is charged per *point* that lands in a bin. A bin with many points pays many
histogram atomics on that one counter; a bin with none pays only the scan/map work.

Non-atomic operations in the same schemes, for completeness:

- `calc_inverse_of_global_sort_idx` (`spreadinterp.hpp:313-325`) and its ghost variant
  (`spread_blockgather_inst.cu:145-166`) write unique slots, so they need no atomic; they depend on
  the histogram's returned rank.
- `fill_ghost_bins` (`spread_blockgather_inst.cu:45-75`) and `ghost_bin_pts_index` (`:77-118`) are
  plain copies into disjoint bins/slots, run as separate kernels after the histogram.
- `calc_subprob`/`map_b_into_subprob` and their `_3d_v1` variants are one-writer-per-slot kernels
  (`spreadinterp.hpp:333-349`, `spread_blockgather_inst.cu:21-43`).
- Method 3's local accumulation is a **non-atomic** shared read-modify-write
  (`spread_output_driven_inst.cu:92`), serialised by one block barrier per point (`:94`).

---

## 10. Binning / sorting schemes, side by side

All four paths use the same family: **linear bin index → per-bin atomic histogram → prefix scan →
inverse permutation scatter**. None uses a comparison sort or a radix sort of keys.

| | Method 1 sorted | Method 1 unsorted | Methods 2, 3 | Method 4 |
|---|---|---|---|---|
| Key | row-major flat fine-bin index, `compute_bin_index` (`spreadinterp.hpp:215-229`) | none | same as method 1 | padded fine-bin index `(bin/m)*binsperobin + bin%m + 1`, expanded by two ghost layers per dim (`blockgather:132-137`) |
| Grouping primitive | global per-bin `atomicAdd` (`spreadinterp.hpp:308`) | identity | same as method 1 (`spread_subprob_inst.cu:144-147`) | global per-bin `atomicAdd` (`blockgather:140`) + **plain count copy** into ghost bins (`:45-75`) |
| Offsets | `thrust::exclusive_scan` (`spread_nupts:116-117`) | n/a | `thrust::exclusive_scan` (`spread_subprob:148-150`) | `thrust::inclusive_scan` (`blockgather:392-397`) |
| Permutation | inverse scatter from `sortidx` (`spread_nupts:120-124`) | `thrust::sequence` (`:126-128`) | inverse scatter (`spread_subprob:151-155`) | inverse scatter into real bins (`blockgather:145-166`) then ghost index copy (`:77-118`) |
| Arrays left behind | `binsize`, `binstartpts`, `sortidx`, `idxnupts` | `idxnupts` only | `binsize`, `binstartpts`, `sortidx`, `idxnupts`, plus subproblem arrays | expanded `binsize`/`binstartpts`, `sortidx`, `idxnupts` (size > `M`), subproblem arrays |
| Bin index computed | twice per point (`:307`, `:321`) | 0 times | twice per point (`spread_subprob:145`, `:152`) | twice per point (`blockgather:133`, `:156`) |

Notes:

- The **only per-bin global atomic histogram outside method 4** is
  `calc_bin_size_noghost` (`spreadinterp.hpp:308`). `calc_inverse_of_global_sort_idx` itself
  contains **no** atomic (ticket wording notwithstanding): it consumes the ranks the histogram
  produced. The histogram and the inverse pass each recompute the bin index, so `fold_rescale` and
  the floor/scale work are paid twice per point in every sorted path.
- Method 4's "ghost" work is an **index-list duplication**, not a histogram: counts are copied,
  entries are copied with a box prefix, so the histogram only ever counts `M` real placements while
  the sorted array can hold up to `2^dim·M` entries (§8.1).
- Method 1 with `gpu_sort=0` is the only spread path whose preprocessing performs **no atomics at
  all** (identity permutation).

---

## 11. Subproblem construction and the local→global merge

### 11.1 Subproblems

- **Methods 2 and 3:** one bin with `occ` points becomes `ceil(occ / maxsubprobsize)` subproblems
  (`spreadinterp.hpp:333-340`). Subproblems of the same bin are contiguous in `subprob_to_bin` and
  contiguous in the sorted point array (`compute_subprob_block_info`,
  `src/cuda/spreadinterp_common.cuh:28-43`). `maxsubprobsize` defaults to 1024
  (`src/cuda/c_interface.cpp:162`) and applies to method 3 as well, despite the docs saying
  method 2 only (`docs/c_gpu.rst:425`).
- **Method 4:** one **outer** bin becomes
  `ceil(total_entries / maxsubprobsize)` subproblems, where `total_entries` includes the ghost
  duplicates (`src/cuda/spread_blockgather_inst.cu:21-34`). Subproblem point slices are contiguous
  in the outer bin's expanded bin range (`:197-203`).

### 11.2 Which cells the merge covers

| Method | Merge region per block | Cells | Overlap with other blocks |
|---|---|---|---|
| 1 | none (direct scatter, no merge) | — | all cells written by any point |
| 2 | padded tile `[offset-ns/2, offset+binsize+ns/2)` per dim, modulo `nf`, minus cells with `idx ≥ nf+ns/2` (`spreadinterp.hpp:269-294`) | `V_bin = Π(binsize_i + 2·ceil(ns/2))` | adjacent bins' collars overlap; subproblems of the same bin overlap completely; periodic wrap can make a cell reachable from both ends |
| 3 | same padded tile, every cell wrapped once modulo `nf`, no `in_region` guard (`spread_output_driven_inst.cu:98-102`, `spreadinterp.hpp:246-262`) | `V_bin` | as method 2 |
| 4 | the outer bin exactly, `[offset, offset+obin)` per dim, no halo, no modulo (`spread_blockgather_inst.cu:264-273`) | `Π obin_i` | only subproblems of the **same** outer bin; outer bins are disjoint because `nf % obinsize == 0` is enforced (`:332-341`) |

### 11.3 Why the merge needs atomics

- **Methods 2/3:** a block writes the whole padded tile, including the `ceil(ns/2)`-cell collar that
  belongs to a neighbouring bin. Every interior output cell is therefore written by up to `2^dim`
  blocks (one per adjacent tile whose collar or interior covers it), and cells near the periodic
  boundary can be written from both ends. Additionally, when `occ > maxsubprobsize`, several blocks
  of the same bin write the *same* full tile, each contributing only its own points and zeros
  elsewhere. Hence a plain store would lose contributions; the code uses a global atomic add.
- **Method 4:** outer bins are a true partition of the output grid and taps are clipped to the
  owning outer bin, so *cross-obin* overlap is zero. The atomics are required **only** because
  several subproblem blocks of the same outer bin each write the complete `Π obin_i` tile
  (`:264-273`); with one block per outer bin the same write could be a store. The shared atomics
  (`:255`) are required because the point loop is point-parallel (`:215`), so two threads can hit the
  same tile cell.
- **Method 1:** no merge; every tap is a global atomic, so no ownership exists at all.

---

## 12. Where the design sketch's three changes would replace existing work

Sketch: `docs/issues/assets/ISSUE-001-destination-owned-nufft-design.md` — Morton key (§5),
sort-based binning with a radix sort instead of a per-bin atomic histogram (§6–§7), output-cell
ownership with non-atomic final stores (§3, §9). The mapping below states which current work items
perform the same job and would therefore be replaced; it is not a design proposal.

| Sketch change | Current code doing that job | What would be replaced / removed |
|---|---|---|
| **Morton key** (§5) | Row-major flat bin key, `compute_bin_index` (`spreadinterp.hpp:215-229`), called from `spreadinterp.hpp:307`, `:321`; method 4's padded-bin key (`blockgather:132-137` and `:155-160`) with `calc_global_index` (`:13-19`) | The key computation itself. Today the key is computed **twice per point** in every sorted path (histogram pass and inverse-permutation pass); a stored Morton key computed once would subsume both. Method 4's `M*(box0+3*box1+9*box2)+i` encoding (`:113-114`, decoded at `:217-225`) is a hand-rolled neighbour/periodic key that a Morton key would subsume. |
| **Sort-based binning** (§6) | `cudaMemsetAsync(binsize)` + `calc_bin_size_noghost` histogram + `thrust::exclusive_scan` + `calc_inverse_of_global_sort_idx`: method 1 `spread_nupts_driven_inst.cu:109-124`; methods 2/3 `spread_subprob_inst.cu:142-155` | The only global per-bin `atomicAdd(int)` in those paths (`spreadinterp.hpp:308`), the `nbins`-sized memset and scan, the `sortidx` array, and the entire inverse-permutation pass. A sort of `(key, point)` produces `idxnupts` directly; `binsize` becomes run lengths and `binstartpts` becomes run boundaries. |
| **Sort-based binning, method 4 variant** (§6–§7) | `locate_nupts_to_bins_ghost` + `fill_ghost_bins` + scan + `calc_inverse_of_global_sort_index_ghost` + `ghost_bin_pts_index` + the host resize: `blockgather:120-166`, `:374-416`; ghost count copy `:45-75`; entry copy `:77-118` | Method 4's histogram and, more importantly, its **materialised ghost index lists** (up to `2^dim·M` entries). Sorted keys plus key-range lookup for a tile's expanded region would replace both the duplication and the rank/scatter passes; the box-code/periodic decode (`:113-114`, `:217-229`) would be replaced by key-range enumeration modulo the grid. |
| **Occupied-bin run construction** (§7) | No direct counterpart. Closest: `calc_subprob` (`spreadinterp.hpp:333-340`), the `numsubprob` scan and `map_b_into_subprob` (`spread_subprob_inst.cu:160-178`), `calc_subprob_3d_v1`/`map_b_into_subprob_3d_v1` (`blockgather:21-43`), and `binstartpts` | A run table (key, begin, count) covers the same information as `binsize`/`binstartpts` and would subsume the subproblem→bin metadata for the sorted grouping. It does **not** subsume load-balancing splits for dense bins (the sketch defers those, §22), which `numsubprob` currently provides. |
| **Output-cell ownership** (§3, §9) | The global merges: method 2 `spread_subprob_inst.cu:95`; method 3 `spread_output_driven_inst.cu:101`; method 4 `spread_blockgather_inst.cu:273`; and, in the limit, method 1's per-interaction scatter `spread_nupts_driven_inst.cu:35/43/56` | These are exactly the global atomics the sketch removes. The final `fw` zeroing (`src/cuda/execute.cu:135`) exists because every path adds into `fw`; unique ownership would replace adds with stores for owned cells. Method 3's local accumulation (`:92`) is already atomic-free and would be kept in spirit; method 4's per-outer-bin tile is already an output partition. |
| **Halo / periodic handling** (§8, §16) | Methods 2/3: `ns/2` collar + single modulo wrap (`spreadinterp.hpp:269-294`, `:246-262`). Method 4: tap clipping to the outer bin (`blockgather:243-244`) plus ghost index copies (`:77-118`) and a `±nf` shift only at the grid boundary (`:217-229`) | The sketch's "expand the destination tile by the kernel halo and enumerate intersecting source bins, mapping coordinates modulo the fine grid" replaces both mechanisms: no collar cells are written outside the owned tile, and no duplicated point entries are materialised. |

Not replaced by the sketch (still needed by any new path): `fold_rescale` and the kernel-support
rule (`include/cufinufft/spreadinterp.hpp:153-196`, `include/cufinufft/utils.hpp:51-55`), separable
kernel evaluation, the `fw` FFT/deconvolution stages, and some form of work cap/load balancing for
dense regions (`gpu_maxsubprobsize` or the sketch's §21.2/§22 extensions).

---

## 13. Where the current code already approximates output-cell ownership

1. **Method 3's per-bin shared tile, with genuinely atomic-free local accumulation.**
   `local_subgrid[idxout] += ...` (`spread_output_driven_inst.cu:92`) is a non-atomic shared
   read-modify-write. It is safe because (a) for one point the flat tap index maps to a unique tile
   cell, so the thread-parallel tap loop has no collisions, and (b) points are processed serially
   with a `__syncthreads()` after each (`:71`, `:94`). This already satisfies the sketch's
   "local accumulation: no atomic updates" half. What is missing for output-cell ownership is the
   global half: the block writes a **padded** tile, so collar cells belong to neighbours; several
   subproblem blocks of the same bin write the same tile; and the tile includes many cells with zero
   contribution.
2. **Method 2's per-bin tile.** Interior cells of a bin are only written by blocks serving that bin,
   which is bin-granularity ownership, but the same collar and multi-subproblem caveats apply, and
   the accumulation is shared-atomic.
3. **Method 4's per-outer-bin tile — the closest existing analogue.**
   The shared buffer is exactly `Π obin_i` with **no padding** (`blockgather:207`, `:211`); taps are
   clipped to `[0, obin)` (`:243-244`); the write-back covers exactly the outer bin's cells with no
   halo and no modulo (`:264-273`); and outer bins partition the fine grid by construction
   (`nf % obinsize == 0`, validated at `:332-341`). So **every output cell already has exactly one
   owning outer bin**, and the only writer blocks are that bin's subproblems. This is
   output-cell ownership at outer-bin granularity, achieved by *duplicating point entries*
   (ghost lists, `:77-118`) instead of letting a block write into its neighbours' halos — the
   sketch's §16 option 2 ("duplicate only boundary-bin descriptors, not point data").
   What still forces atomics:
   - **multiple subproblem blocks per outer bin** (`calc_subprob_3d_v1:21-34`; grid at `:293`), each
     writing the whole `Π obin_i` tile even though it only touched some cells — the sole reason for
     `blockgather:273`;
   - **point-parallel shared accumulation** inside a block (`:215`, atomic at `:255`);
   - **ghost-entry inflation** of the index list (up to `2^dim·M` entries, §8.1), which is a
     preprocessing/memory cost rather than an atomic cost.
4. **Method 1 has no output ownership.** Sorted method 1 only changes the *order* in which points
   issue their global atomics; locality is improved, exclusivity is not.
5. **`fw` is always add-accumulated.** `execute` zeroes the whole fine grid
   (`src/cuda/execute.cu:135`) and all four paths add into it; there is no cell whose final value is
   written by a single store today.

---

## 14. Cost classification: per-point vs per-interaction, preprocessing vs spread

Notation: `V_bin = Π_i (binsize_i + 2·ceil(ns/2))`; for method 4 `V_obin = Π_i obin_i` and
`E = |idxnupts| ≈ ρ·M` with `ρ ∈ [1, 2^dim]` the ghost replication factor. `occ` = points in a bin;
`maxsubprobsize = 1024` by default.

### Per-point and cheap, preprocessing (once per `setpts`, independent of `ntransf`)

| Cost | Method 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| global `atomicAdd(int)` | 1 (if `gpu_sort`) | 1 | 1 | 1 |
| bin-key computation (`fold_rescale` × dim + floor) | 2× | 2× | 2× | 2× (`+` box/expand arithmetic) |
| bin-index-dependent writes | `sortidx`, `idxnupts` | same | same | `sortidx`, `idxnupts`, ghost copies |
| per-entry extra work | — | — | — | `O(E)` copies + `E`-sized array |
| per-bin work | memset + exclusive scan of `nbins` | same + `calc_subprob` + `map_b_into_subprob` + inclusive scan of `nbins` | same | memset/scan of expanded bins + ghost copies + per-obin scan |
| host sync | none | 1 (`spread_subprob:170-174`) | 1 | 1 (`blockgather:434-438`) |

### Per-interaction and per-tile, spread (per transform, × `blksize` launches)

| Method | shared atomics | local non-atomic RMW | global atomics | barriers |
|---|---|---|---|---|
| 1 | 0 | 0 | `ns^dim` per point (1 complex atomic each) | 0 |
| 2 | `ns^dim` per point (2 hardware atomics each) | 0 | `V_bin` per subproblem = `V_bin·ceil(occ/maxsub)` per bin | 2 per block |
| 3 | 0 | `ns^dim` per point | `V_bin` per subproblem | 2 per block **+ 1 per point** (`:94`) |
| 4 | up to `ns^3` per entry (`E` entries; clipped at tile edges, so ≈ `ns^3` per point in total) | 0 | `V_obin` per subproblem = `V_obin·ceil(entries/maxsub)` per outer bin | 2 per block |

Observations that follow from the code, stated as facts:

- Method 1 is the only path with **no preprocessing→spread amortisation of the grid write**: its
  global atomic count is exactly `M·ns^dim` per transform regardless of bin occupancy.
- Methods 2/3/4 convert most of that into **shared** atomics or shared RMWs, plus a global merge
  whose per-point cost is `V_tile / points_per_block`: minimal for densely occupied bins
  (`V_tile/maxsubprobsize` per point) and maximal for nearly empty bins (`V_tile` per point). For
  orientation, 3-D `ns=6` gives `ceil(ns/2)=3`, so `V_bin = (b+6)^3` (e.g. `b=8 → 2744`,
  `b=23 → 24389`) and `ns^3 = 216` interactions per point; method 4's default `obin=8` gives
  `V_obin = 512`.
- Method 3 replaces method 2's `ns^dim` shared atomics with `ns^dim` non-atomic shared updates but
  adds **one block-wide barrier per point** (`spread_output_driven_inst.cu:94`), so its per-point
  cost scales with the number of points per block (up to `maxsubprobsize`), not just with `np`.
- Method 4's ghost replication inflates `E` and therefore indexing/staging work, but not the number
  of mathematical interactions, because each outer bin clips taps to its own cells.
- All per-transform numbers are multiplied by `blksize` (one kernel launch per transform) inside a
  single `spreadSorted` call; preprocessing is not repeated per transform.

---

## 15. Gotchas relevant to any later comparison

- **`gpu_maxsubprobsize` also governs method 3** (`src/cuda/spreadinterp_common.cuh:28-43`;
  `spread_output_driven_inst.cu:27-31`) although `docs/c_gpu.rst:425` says method 2 only. Method 3's
  block count, point range, and per-point barrier count follow it.
- **`gpu_np` affects only method 3's shared-memory split** and is zeroed for all other methods
  (`src/cuda/makeplan.cu:270`), so method 2's tile volume is limited only by
  `shared_memory_required()` with `np = 0`.
- **Method 4 is capped at 48 KB of shared memory per block** with a hard-coded literal
  (`spread_blockgather_inst.cu:284`) and the dynamic-shared-memory opt-in call is commented out
  (`:291`), so it cannot use the full 64 KB/228 KB that the other methods can request.
- **Method 4 duplicates point indices**: `idxnupts` is resized to the total number of *entries*
  (`:399-403`), up to `2^dim·M` by the ghost construction for the default `m = obin/bin = 2`.
  `p.M` is used as the base of the box encoding (`:113-114`, `:217-225`), so the encoding assumes
  `M·26` fits in the `int` used by `idxnupts`.
- **Method 4 ghost reach**: only the immediately adjacent fine bin per dimension is copied
  (`:61-68`, `:97-108`). Combined with the support rule (`start = ceil(x - ns/2)`,
  `end = start + ns - 1`), this assumes a bin width of at least `ceil(ns/2)-1` so that no point
  further away can reach the tile; nothing validates that relationship.
- **Method 2 has an `in_region` guard, method 3 does not.** `shared_mem_copy_helper` skips padded
  cells with `idx >= nf + ns/2` (`spreadinterp.hpp:285`), whereas method 3's write-back wraps every
  padded cell once (`spreadinterp.hpp:255-256`). The two merges can therefore differ in which
  out-of-range cells they touch when `binsize` does not divide `nf`.
- **`compute_bin_index` clamps** the top/negative edges into the first/last bin
  (`spreadinterp.hpp:223-224`), a behaviour a Morton key would have to reproduce (relevant to
  ISSUE-004's contract, not a design statement here).
- **Method 1 unsorted has zero preprocessing atomics** and an identity permutation (`:126-128`);
  the other three paths always pay the histogram.
- **Interpolation is not a mirror**: only methods 1 and 2 have interp kernels
  (`src/cuda/spreadinterp.cpp:50-62`); methods 3 and 4 are spread-only. Neither interp path uses
  atomics (grep over `interp_*_inst.cu`: no `atomicAdd`).
- **All preprocessing atomics are once per `setpts`**, so a baseline that times `execute()` only
  will not see them; a baseline that times `setpts`+`execute` or repeatedly calls `setpts` will.

---

## 16. Source index (primary)

Code:

- `include/cufinufft/spreadinterp.hpp` — shared device helpers, `compute_bin_index`, padded-tile
  geometry, shared→global copy helper, `calc_bin_size_noghost`, `calc_inverse_of_global_sort_idx`,
  `calc_subprob`, `map_b_into_subprob`.
- `include/cufinufft/spreadinterp.hpp:153-196` — per-point kernel/start computation (both variants).
- `include/cufinufft/utils.hpp:56-102` — atomic wrappers and the sm_90 float2 path.
- `include/cufinufft/cufinufft_plan_t.hpp:357-374, 565-580` — array meanings and device view.
- `include/cufinufft/heuristics.hpp:41-54` — `cufinufft_set_shared_memory` opt-in.
- `src/cuda/spread_nupts_driven_inst.cu` — method 1.
- `src/cuda/spread_subprob_inst.cu` — method 2 and the shared method-2/3 preprocessing.
- `src/cuda/spread_output_driven_inst.cu` — method 3.
- `src/cuda/spread_blockgather_inst.cu` — method 4.
- `src/cuda/spreadinterp_common.cuh` — block↔subproblem mapping, host bin layout.
- `src/cuda/spreadinterp.cpp` — stage dispatch.
- `src/cuda/makeplan.cu:127-209` — array allocation per method; `:270` — `gpu_np` zeroing;
  `:306-320` — auto-method.
- `src/cuda/setpts.cu:65-76, 300-317` — where preprocessing is invoked.
- `src/cuda/execute.cu:135-138, 238-246` — `fw` zeroing + spread invocation.
- `src/cuda/heuristics.cu:93-126` — shared-memory accounting; `:290-352` — bin-size selection;
  `:146-244` — method-3 `(bin, np)` tables.
- `src/cuda/c_interface.cpp:158-170` — option defaults.

In-repo documentation used for intent (secondary to the code):

- `docs/impl_gpu.rst:19-90` — output-driven description, including "removes the need for
  `AtomicAdd` on the local subgrid" (`:81`) and the unchanged global atomic merge (`:83-86`).
- `docs/c_gpu.rst:407-433` — option semantics, including the `gpu_maxsubprobsize` method-2-only
  statement (`:425`).
- `docs/issues/assets/ISSUE-001-destination-owned-nufft-design.md` — the design sketch's Morton key
  (§5), sort-based binning (§6–§7), output-cell ownership (§3, §9), and periodic strategy (§16).
