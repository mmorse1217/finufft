# ISSUE-004 research: spread correctness contract for an output-owned 3D type-1 kernel

**Ticket:** [`ISSUE-004`](../done/ISSUE-004-spread-correctness-contract.md)  
**Source snapshot:** commit `ee272845` of this worktree.  
**Method:** source reading only; no build, GPU run, or measurement was performed for this note.  
**Scope:** the semantics a new output-owned 3D type-1 spread kernel must reproduce. This is a
fact note: it describes the existing cuFINUFFT reference path and a comparison oracle. It
does not propose an implementation or change library code.

## Bottom line

- The reference must be **cuFINUFFT's GPU paths**, not CPU FINUFFT. Current cuFINUFFT uses
  the ES kernel (`exp(beta*(sqrt(1-z^2)-1))`); current CPU FINUFFT uses the PSWF kernel and
  a different Horner table. They are not interchangeable for a "same kernel, different
  summation order" comparison. See [`kernel.cpp:49-54`](../../../src/common/kernel.cpp) and
  [`spreadinterp.hpp:91-134`](../../../include/cufinufft/spreadinterp.hpp).
- For type 1, the natural baseline is `gpu_method=3` (output-driven / OD) with
  `gpu_spreadinterponly=1`. The default method auto-choice is **method 2**, not 3, for
  type 1, so the method must be pinned explicitly
  ([`makeplan.cu:304-310`](../../../src/cuda/makeplan.cu)).
- The spread itself is a sum over the same multiset of `strength * phi_x * phi_y * phi_z`
  terms. A different tile/ownership scheme may change only the order of the additions and
  the grouping of the three real factors. Bitwise equality is not expected; relative
  roundoff is.
- Coordinate folding and output-cell wrapping are two separate operations. Folding maps
  the user coordinate into `[0,nf)`; wrapping maps a support index outside the fine grid
  back into `[0,nf)`.
- For even `ns`, the OD tile's left and right halo are both `ns/2` and every allocated halo
  cell is in the used range. For odd `ns`, the allocated halo is `ceil(ns/2)` on both
  sides, but the leftmost allocated cell is never written; the used left overhang is
  `floor(ns/2)` and the used right overhang is `ceil(ns/2)`.
- `gpu_spreadinterponly=1` makes the fine grid exactly the user mode box (`nf=N`) and skips
  FFT/deconvolution. `modeord` belongs to deconvolution only; it has no effect on the spread
  stage or on raw spread-only output.
- The oracle is relative L2 over **all** fine-grid cells, against a pinned cuFINUFFT
  spread-only run with the same `ns`, `beta`, kernel evaluator, folded coordinates and
  strengths. Calibrate the pass threshold from the reference's own atomic/summation
  nondeterminism; for FP32, pure reordering is expected around `1e-7..1e-5` relative L2,
  for FP64 around `1e-16..1e-14`, not at the requested NUFFT tolerance.

## 1. Coordinate mapping: `fold_rescale`, start index, support

### 1.1 The formula

The CPU helper is

```text
t = fma(x, 1/(2*pi), 0.5)
r = (t - floor(t)) * N
```

([`simd.hpp:321-328`](../../../include/finufft/simd.hpp)). Mathematically this is
`r = ((x + pi) mod 2*pi) * N/(2*pi)`, i.e. the user coordinate mapped to fine-grid index
units, with `x=-pi` at index 0.

The cuFINUFFT device helper has the same formula but uses directed rounding on device:

- `result = cudaFMA(x, x2pi, 0.5)`;
- float: `__fsub_rd(result, floorf(result))`, then `__fmul_rd(..., N)`;
- double: `__dsub_rd(result, floor(result))`, then `__dmul_rd(..., N)`;
- the `__host__` fallback uses ordinary `std::fma`, subtraction and multiplication.

See [`spreadinterp.hpp:57-89`](../../../include/cufinufft/spreadinterp.hpp). The constant
`x2pi` is the same double as the CPU `INV_2PI`
([`constants.h:19-21`](../../../include/finufft_common/constants.h)), but the device
directed-rounding steps mean the last bits of `r` can differ from a host computation. A
kernel that must match cuFINUFFT should reuse the device folding routine (or reproduce its
rounding), not recompute the fold on the host.

### 1.2 Range, negative, edge and out-of-domain coordinates

- The result is mathematically in `[0,N)` for finite `x`. The transformation handles
  negative and multi-period inputs because it uses `floor`, not a single conditional.
  See [`simd.hpp:316-319`](../../../include/finufft/simd.hpp) and the CPU docs at
  [`spreadinterp.hpp:238-243`](../../../include/finufft/spreadinterp.hpp).
- Large `|x|` is accepted, but folding loses precision as `|x|` grows
  ([`simd.hpp:316-319`](../../../include/finufft/simd.hpp)).
- Examples for period `2*pi`:
  - `x=-3*pi -> 0`, `x=-pi -> 0`, `x=pi -> 0`, `x=3*pi -> 0`;
  - `x=-pi-eps ->` just below `N` (i.e. wraps to the top of the grid);
  - `x=pi-eps ->` just below `N`;
  - `x=0 -> N/2`; this is an integer if `N` is even, but a half-integer if `N` is odd.
- `r=N` is possible from roundoff for `x` near `+pi`; the CPU bin-sort comment explicitly
  allows this ([`spread.hpp:512-515`](../../../include/finufft/spread.hpp)). The existing GPU
  cell wrap handles an index at or above `N` by subtracting `N` once
  ([`spreadinterp.hpp:253-257`](../../../include/cufinufft/spreadinterp.hpp)). The
  `cufinufft3d_test` deliberately uses points in `[-3*pi,3*pi)` to exercise the fold
  ([`cufinufft3d_test.cu:42-47`](../../../test/cuda/cufinufft3d_test.cu)).

### 1.3 The start rule and the support samples

Given folded coordinate `r`, both CPU and GPU use the same leftmost-index rule:

```text
s = ceil(r - ns/2)
u = s - r
```

Then for `j = 0 .. ns-1`:

```text
kernel argument = u + j = (s + j) - r
kernel weight   = phi(u + j)
target cell     = (s + j) mod nf
```

GPU: [`spreadinterp.hpp:154-172`](../../../include/cufinufft/spreadinterp.hpp). CPU 3D:
[`spread.hpp:303-384`](../../../include/finufft/spread.hpp), especially the three `ceil`
lines at `328-333`. CPU 1D/2D follow the same rule
([`spread.hpp:110-124`](../../../include/finufft/spread.hpp),
[`spread.hpp:232-238`](../../../include/finufft/spread.hpp)).

The exact-arithmetic invariant is

```text
-ns/2 <= u < -ns/2 + 1
```

so the `ns` samples are the `ns` integer grid points whose distances from `r` lie in the
kernel support. `u` can be exactly `-ns/2`:

- `ns` even: when `r` is an integer;
- `ns` odd: when `r` is a half-integer (for example `x=0` with an odd `nf`).

For CPU's 1D SIMD subproblem path, `u` is explicitly clamped to
`[-ns/2, -ns/2+1]` to protect the polynomial panel domain when `N*eps_mach` is large
([`spread.hpp:113-121`](../../../include/finufft/spread.hpp)); the CPU 2D/3D kernels do not
repeat that clamp ([`spread.hpp:235-236`](../../../include/finufft/spread.hpp),
[`spread.hpp:331-333`](../../../include/finufft/spread.hpp)), and CUDA `get_kerval...` does
not clamp at all. The CUDA version relies on `ceil` giving a value in range in exact
arithmetic. A matching CUDA kernel should follow CUDA, not add the CPU 1D clamp.

### 1.4 Even and odd `ns`

The kernel support is always `[-ns/2, ns/2]`; the implementation takes exactly `ns` samples
starting at `ceil(r - ns/2)`, not `ns+1` samples.

| `ns` | support half-width | samples near an integer-aligned `r` | notes |
|---|---|---|---|
| even, e.g. 6 | integer `3` | `r-3, r-2, ..., r+2` | for integer `r`, left support endpoint included; right endpoint excluded |
| odd, e.g. 5 | half-integer `2.5` | `r-2, r-1, r, r+1, r+2` | interior for integer `r`; left endpoint can occur for half-integer `r` |

For even `ns` and `r=0` with `ns=6`, `s=-3` and the samples are
`-3,-2,-1,0,1,2`; for odd `ns=5`, `s=-2` and the samples are `-2,-1,0,1,2`. The rule is the
same; only the half-width parity changes.

### 1.5 Endpoint conventions differ by evaluator

This matters when a sample lands exactly on a support endpoint. In exact arithmetic the start
rule can put a sample at the left endpoint `-ns/2`; the right endpoint is not normally
sampled, but the evaluator conventions at the right endpoint are listed for completeness.

- CPU's scalar `evaluate_kernel_runtime` selects a panel with
  `x > -ns/2+i && x <= -ns/2+i+1`, so it returns 0 at the left endpoint and uses the
  right endpoint ([`spreadinterp.hpp:85-91`](../../../include/finufft/spreadinterp.hpp)).
- CUDA direct `evaluate_kernel` returns 0 when `z^2 >= 1`, so it is zero at both endpoints
  ([`spreadinterp.hpp:91-102`](../../../include/cufinufft/spreadinterp.hpp)).
- CUDA Horner `eval_kernel_vec_horner` evaluates the generated polynomial with no endpoint
  test, so an endpoint sample gets the polynomial's endpoint value (not constrained to be
  zero), and a rounding-induced out-of-range `u` is extrapolated rather than zeroed
  ([`spreadinterp.hpp:113-134`](../../../include/cufinufft/spreadinterp.hpp)).
- CPU's vector evaluator also evaluates the panel polynomial with no endpoint test
  ([`simd.hpp:360-432`](../../../include/finufft/simd.hpp)); CPU's scalar runtime path is a
  different function.

Consequence: do not replace the reference evaluator with a "mathematically equivalent"
`exp`/`sqrt` or PSWF. For grid-aligned or half-aligned points, endpoint behavior is part of
the observed output. Add exact grid points to the oracle data so this is exposed.

## 2. Kernel evaluation, `sigma`, `beta`, and `ns`

### 2.1 Two cuFINUFFT evaluators, selected by `gpu_kerevalmeth`

`gpu_kerevalmeth` is a public option, default **1**
([`c_interface.cpp:157-161`](../../../src/cuda/c_interface.cpp),
[`cufinufft_opts.h:23`](../../../include/cufinufft_opts.h)).

**`gpu_kerevalmeth=0`: direct ES evaluator.**
For `z = 2*x/ns`:

```text
phi(x) = exp(beta * (sqrt(1 - z^2) - 1))   if z^2 < 1
       = 0                                  otherwise
```

with `es_c = (2/ns)^2`, so `z^2 = es_c*x*x`. `eval_kernel_vec` evaluates
`phi(|x+i|)` for `i=0..ns-1`
([`spreadinterp.hpp:91-111`](../../../include/cufinufft/spreadinterp.hpp)). The absolute
value is valid because the ES kernel is even. This path accepts nonstandard
`upsampfac > 1` ([`makeplan.cu:51-66`](../../../src/cuda/makeplan.cu),
[`c_gpu.rst:421-423`](../../c_gpu.rst)).

**`gpu_kerevalmeth=1`: Horner piecewise polynomial.**
For `z = 2*u + ns - 1`, generated tables fill `ker[i]` with an approximation to the ES
kernel at `u+i`, `i=0..ns-1`
([`spreadinterp.hpp:113-134`](../../../include/cufinufft/spreadinterp.hpp)). The coefficient
tables are the generated includes
[`ker_horner_allw_loop.inc`](../../../include/cufinufft/contrib/ker_horner_allw_loop.inc)
(for `upsampfac=2.0`) and
[`ker_lowupsampfac_horner_allw_loop.inc`](../../../include/cufinufft/contrib/ker_lowupsampfac_horner_allw_loop.inc)
(for `1.25`). Horner is selected by exact floating-point equality `upsampfac == 2.0` or
`== 1.25`; a nonstandard sigma with `kerevalmeth=1` is rejected at plan creation
([`makeplan.cu:51-58`](../../../src/cuda/makeplan.cu)).

The direct and Horner paths are **not** the same function sample-for-sample: Horner is an
approximation, and endpoint values differ as described above. They produce different
fine-grid values even with identical points and strengths. Pin one evaluator for the
reference and use the same one in the new kernel; a difference between them is not
"summation order."

**CPU is a third evaluator.** Current CPU FINUFFT always uses Horner (the
`kerevalmeth`/`kerpad` knobs are documented as ignored at
[`spreadinterp.hpp:8-13`](../../../include/finufft/spreadinterp.hpp)), but the fitted kernel
is PSWF (`kerformula` default 8, [`makeplan.hpp:150-151`](../../../include/finufft/makeplan.hpp),
[`kernel.cpp:49-54`](../../../src/common/kernel.cpp)). CPU's panel map and coefficient
layout are in [`makeplan.hpp:203-300`](../../../include/finufft/makeplan.hpp). Its
`evaluate_kernel_vector` uses `z = fma(2, x, ns-1)` and the precomputed table
([`simd.hpp:330-432`](../../../include/finufft/simd.hpp)), but the underlying kernel is not
the GPU ES kernel. Do not use a CPU FINUFFT spread result as the reference for a
cuFINUFFT-matching kernel.

### 2.2 `sigma` and `beta` as cuFINUFFT sets them

On the GPU path, `setup_spreadinterp()` is the source of truth
([`makeplan.cu:34-108`](../../../src/cuda/makeplan.cu)). Its values differ from current CPU
FINUFFT, which uses the PSWF `kerformula=8`,
`beta_cutoff = pi*ns*(1 - 1/(2*sigma))` and `beta = beta_cutoff - 0.05`
([`kernel.cpp:95-115`](../../../src/common/kernel.cpp)), with a different `ns` selection
law and clamp ([`makeplan.hpp:164-188`](../../../include/finufft/makeplan.hpp)):

- **sigma (`upsampfac`)**: `opts.upsampfac == 0` auto-selects `2.0`, except type 3 with
  `tol >= 1e-9` selects `1.25`. For type 1, auto is therefore `2.0`
  ([`makeplan.cu:283-293`](../../../src/cuda/makeplan.cu)). The default option is `0.0`
  ([`c_interface.cpp:161`](../../../src/cuda/c_interface.cpp)).
- **`ns` for sigma 2.0**:
  `ns = ceil(-log10(eps/10))`, then `max(2, ns)`, then clipped to
  `MAX_NSPREAD<T>` (`12` for float, `16` for double)
  ([`makeplan.cu:83-95`](../../../src/cuda/makeplan.cu),
  [`constants.h:10-14`](../../../include/finufft_common/constants.h)). An `eps` below
  machine epsilon is first raised to machine epsilon
  ([`makeplan.cu:73-79`](../../../src/cuda/makeplan.cu)). For example,
  `eps=1e-5` gives `ns=6`; `eps` in `(1e-6, 1e-5]` also gives `ns=6`.
- **`ns` for a nonstandard sigma**:
  `ns = ceil(-ln(eps) / (pi * sqrt(1 - 1/sigma)))`
  ([`makeplan.cu:84-86`](../../../src/cuda/makeplan.cu)).
- **`beta`**: `beta/ns = 2.30` by default; exceptions are `ns=2 -> 2.20`,
  `ns=3 -> 2.26`, `ns=4 -> 2.38`. For `sigma != 2.0`,
  `beta/ns = gamma*pi*(1 - 1/(2*sigma))` with `gamma=0.97`. Then `beta = (beta/ns)*ns`
  ([`makeplan.cu:97-105`](../../../src/cuda/makeplan.cu)). The same numbers are hardwired in
  the Horner-table generator ([`get_degree_and_beta.m:23-32`](../../../devel/get_degree_and_beta.m)).
- **Derived GPU values**: `es_c = 4/ns^2`, `es_beta = spopts.beta`, `sigma =
  spopts.upsampfac` ([`cufinufft_plan_t.hpp:530-598`](../../../include/cufinufft/cufinufft_plan_t.hpp)).
  The direct path uses `es_c` and `es_beta`; the Horner path uses `sigma` to select the
  table, not `beta`.

For the prototype target `ns=6`, `sigma=2.0`: `beta = 2.30*6 = 13.8`;
`es_c = 4/36 = 1/9`. A new kernel should read `ns`, `beta`, `es_c` and `sigma` from the
same plan/configuration rather than re-deriving them, so a later nonstandard-sigma or
different-tol case cannot silently diverge.

## 3. Fine-grid layout, padding, halo and periodic edges

### 3.1 Flat layout

The fine grid `fw` is complex, interleaved, and **x fastest, then y, then z slowest**:

```text
fw[ix + nf1*(iy + nf2*iz)]
```

The same index convention applies to the mode array `fk` with `N1,N2,N3`. This is explicit
in `deconv_nd`, where `nf_acc = {1, nf1, nf1*nf2}` and `fwidx += wn * nf_acc[idim]`
([`execute.cu:28-55`](../../../src/cuda/execute.cu)). cuFFT is planned with the dimensions
reversed (`n[idim] = nf123[dim-idim-1]`) to be consistent with that x-fastest layout
([`makeplan.cu:366-376`](../../../src/cuda/makeplan.cu)). The CPU docs state the same
x-fastest ordering ([`spreadinterp.hpp:232-237`](../../../include/finufft/spreadinterp.hpp)).

For a normal type-1 NUFFT, each fine-grid dimension is chosen by `set_nf_type12`:
`ceil(sigma * N)`, floored at `2*ns`, then rounded up to an even 2,3,5-smooth size
([`fft.cu:166-179`](../../../src/cuda/fft.cu)). For spread-only mode, `nf123 = mstu` exactly
([`makeplan.cu:336-353`](../../../src/cuda/makeplan.cu)).

### 3.2 Method 3's per-bin tile

For `gpu_method=3`, preprocessing builds `idxnupts` and the subproblem metadata from bins of
size `gpu_binsize{x,y,z}` ([`spread_subprob_inst.cu:136-181`](../../../src/cuda/spread_subprob_inst.cu)).
The number of bins is `ceil(nf_i / binsize_i)` ([`spreadinterp.hpp:136-145`](../../../include/cufinufft/spreadinterp.hpp)).
A bin's global lower corner is `offset_i = bin_index_i * binsize_i`
([`spreadinterp.hpp:198-210`](../../../include/cufinufft/spreadinterp.hpp)).

The local shared tile is padded by `h = ceil(ns/2)` fine-grid cells on **each** side:

```text
h          = (ns + 1) / 2          // integer division
rounded_ns = 2*h                   // equals ns if ns even, ns+1 if ns odd
padded_size_i = binsize_i + rounded_ns
```

([`spreadinterp.hpp:231-242`](../../../include/cufinufft/spreadinterp.hpp);
`rounded_ns` is written as `((ns + 1) / 2) * 2` and `ns_2` as `(ns + 1) / 2`).
The length uses the configured `binsize`, not the actual remaining width of the last bin,
so the last bin's tile extends past `nf`; the overrun is normally about `h` cells and is
larger when `nf` is not a multiple of `binsize`. Those unused cells wrap.

For a point assigned to a bin, the local start is `start_i = s_i - offset_i`. The kernel
writes to local index

```text
start_i + h + x0,   x0 = 0..ns-1
```

and the local-to-global map is `global_i = local_i + offset_i - h`, with one
`+nf_i`/`-nf_i` conditional for wrap
([`spread_output_driven_inst.cu:71-92`](../../../src/cuda/spread_output_driven_inst.cu),
[`spreadinterp.hpp:244-262`](../../../include/cufinufft/spreadinterp.hpp)). Therefore the
point's global support in one dimension is exactly the `ns` cells `s_i .. s_i+ns-1`, as
required.

A bin whose occupancy exceeds `gpu_maxsubprobsize` is split into several subproblem blocks
(`compute_subprob_block_info`, [`spreadinterp_common.cuh:28-43`](../../../src/cuda/spreadinterp_common.cuh)).
Each block still zeroes and merges the **entire** padded tile, not a sub-tile; a block's
points may cover only part of it and the rest is added as zero. So one bin can atomically
merge the same global tile several times. That changes the summation order but not the
spatial footprint described here.

### 3.3 Exactly which cells the OD subgrid covers

Let `floor(ns/2)` mean integer division and `ceil(ns/2) = h`.

- **Allocated local tile**, per dimension:
  `local 0 .. binsize+2h-1`.
- **Allocated global footprint**, per dimension:
  `offset - h .. offset + binsize - 1 + h`.
- **Used global range** (union over points whose folded coordinate lies in
  `[offset, offset+binsize)`):
  `offset - floor(ns/2) .. offset + binsize - 1 + ceil(ns/2)`.

The used range follows from `s_min = ceil(offset - ns/2) = offset - floor(ns/2)` and
`s_max = ceil((offset+binsize)^- - ns/2) = offset + binsize - floor(ns/2)`; the support
extends `ns-1` beyond `s`. Equivalently, in local coordinates:

| `ns` | `h` | local tile length | used local range | unused allocated cells |
|---:|---:|---:|---|---|
| 5 | 3 | `binsize+6` | `1 .. binsize+5` | local 0 |
| 6 | 3 | `binsize+6` | `0 .. binsize+5` | none |
| 7 | 4 | `binsize+8` | `1 .. binsize+7` | local 0 |
| 8 | 4 | `binsize+8` | `0 .. binsize+7` | none |

Thus:

- **Even `ns`**: `floor(ns/2) = ceil(ns/2) = ns/2`; the used range equals the allocated
  footprint. The kernel can touch both halo edges.
- **Odd `ns`**: the left allocated halo is `ceil(ns/2)`, but the leftmost allocated cell is
  one cell farther left than any point in the bin can reach; it stays zero. The used left
  overhang is `floor(ns/2)`, and the used right overhang is `ceil(ns/2)`. OD still zeroes
  and atomically merges that leftmost cell, adding zero.

A point assigned to a bin can therefore deposit into cells that belong to a neighbouring
bin (and, at the periodic edges, to the opposite side of the grid). The bin assignment is
only for scheduling; the contribution set is defined by the point's support `s_i..s_i+ns-1`,
not by the bin index. This is why OD's global merge is atomic. An output-owned kernel must
consider every point whose support covers an owned cell, not only points whose binned tile
is the owned tile.

Concrete even-`ns=6`, `binsize=8`, `nf=24` example:

- point at `r=0` in bin 0 (`offset=0`): `s=-3`, writes cells `-3..2`, wrapping to
  `21,22,23,0,1,2`;
- point at `r=8-eps` still in bin 0: `s=5`, writes `5..10`;
- point at `r=24-eps` in the last bin (`offset=16`): `s=21`, writes `21..26`, wrapping to
  `21,22,23,0,1,2`;
- point at exactly `r=24` (possible from fold roundoff) is clamped to the last bin by
  `compute_bin_index` and also wraps its support once
  ([`spreadinterp.hpp:215-229`](../../../include/cufinufft/spreadinterp.hpp)).

The local tile is zeroed before accumulation
([`spread_output_driven_inst.cu:52-57`](../../../src/cuda/spread_output_driven_inst.cu)),
but the **global** `fw` is not zeroed by the kernel; `execute_type1` memsets
`blksize*nf` complex values before calling `spreadSorted`
([`execute.cu:122-138`](../../../src/cuda/execute.cu)). A new kernel that writes only owned
cells must either be guaranteed that every cell is written (including cells with no
contributions) or zero the output itself; a kernel that only touches contributing cells
needs the same pre-zero contract.

### 3.4 Periodic edges and the one-wrap assumption

The local-to-global map uses a single conditional, not `%`:

```text
idx = local + offset - h
if idx < 0:      idx += nf
if idx >= nf:    idx -= nf
```

([`spreadinterp.hpp:253-257`](../../../include/cufinufft/spreadinterp.hpp)). This is enough
for the actual support of a point because `r` is folded into `[0,nf)`: the support extends
at most `floor(ns/2)` cells left or `ceil(ns/2)` cells right of the period boundaries, and
`nf >= 2*ns` in normal plans. It assumes a tile does not span more than one full period
beyond the grid. A new kernel using modulo `nf` for every cell is equivalent for valid
inputs and safer for arbitrary tile placement; it must still reproduce the same contribution
sets.

The CPU path does the same wrap differently: it sizes a data-dependent subgrid from the
minimum and maximum folded point, then `add_wrapped_subgrid` splits x into below-zero and
above-`N` pieces and fills y/z wrapped index lists
([`spread.hpp:791-868`](../../../include/finufft/spread.hpp),
[`spread.hpp:386-451`](../../../include/finufft/spread.hpp)). Its subgrid is
`offset_i = ceil(min_i - ns/2)` and
`size_i = ceil(max_i - ns/2) - offset_i + ns`; it covers exactly
`offset_i .. offset_i+size_i-1`. The extra `padded_size1` in the CPU allocator is only SIMD
write padding: `add_wrapped_subgrid` advances the input by `padded_size1` but copies only
`size1` cells ([`spread.hpp:435-448`](../../../include/finufft/spread.hpp)). Do not infer the
OD halo from the CPU subgrid; they are different staging strategies.

## 4. Strengths, `idxnupts`, `setpts`, and `ntr`

### 4.1 `setpts` keeps raw point pointers; folding happens at execution

For type 1/2, `setpts_type12` (a) stores `M`, (b) allocates `idxnupts`/`sortidx`, (c) sets
`kxyz = {d_kx, d_ky, d_kz}`, and (d) calls `indexSort()`
([`setpts.cu:27-77`](../../../src/cuda/setpts.cu)). It does **not** copy or rescale the
coordinates. The doc block above it mentions rescaling
([`setpts.cu:30-39`](../../../src/cuda/setpts.cu)), but the code path stores the user
pointers directly; the kernel folds them on every execution. The plan retains the pointers,
so the caller must keep the point arrays valid until execute.

### 4.2 `idxnupts` is the bin-sort permutation

For method 3, `do_indexSort_subprob_and_OD`:

1. computes each point's bin with `compute_bin_index` (`fold_rescale` then
   `floor(r/binsize)`, clamped to a valid bin);
2. atomically histograms bin sizes (`calc_bin_size_noghost`);
3. exclusive-scans to `binstartpts`;
4. writes the inverse permutation
   `idxnupts[bin_startpts[bin] + sortidx[i]] = i`.

See [`spread_subprob_inst.cu:136-155`](../../../src/cuda/spread_subprob_inst.cu),
[`spreadinterp.hpp:300-325`](../../../include/cufinufft/spreadinterp.hpp), and
[`spreadinterp.hpp:215-229`](../../../include/cufinufft/spreadinterp.hpp). The bin sorting
does not change the set of contributions; it only changes their order. The output-driven
kernel reads `nuptsidx = idxnupts[ptstart+i]`, reads the strength as `c[nuptsidx]`, and uses
the point coordinates at index `nuptsidx`
([`spread_output_driven_inst.cu:61-65`](../../../src/cuda/spread_output_driven_inst.cu)).
A new kernel may use a different ordering, but for each original point `j` it must use
`c[j]` and coordinates `x[j],y[j],z[j]`.

### 4.3 `ntr` / `ntransf` batching

The user transform index is the **slowest** array dimension. For type 1:

- strengths for transform `t` are at `c + t*M`;
- the output fine grid for transform `t` is at `fw + t*nf`;
- execute loops in batches of `batchsize`, with `blksize = min(batchsize, remaining)`,
  memsets each `blksize*nf` output block, calls `spreadSorted(c, fw, blksize)`, and skips
  FFT/deconvolution when spread-only
  ([`execute.cu:105-149`](../../../src/cuda/execute.cu));
- the method-3 driver loops `t=0..blksize-1` and passes `c+t*M`, `fw+t*nf` to the kernel
  ([`spread_output_driven_inst.cu:105-128`](../../../src/cuda/spread_output_driven_inst.cu)).

`idxnupts` is built once per `setpts` and reused for every transform. The new kernel must
preserve the per-transform strides and the `c[j]` pairing; in the prototype,
`ntransf=1` removes this variable.

## 5. `gpu_spreadinterponly` and `modeord`

### 5.1 `gpu_spreadinterponly=1`

The flag defaults to 0 ([`c_interface.cpp:155`](../../../src/cuda/c_interface.cpp)). It is
only meaningful for types 1 and 2; type 3's outer plan forces it on internally
([`makeplan.cu:277-279`](../../../src/cuda/makeplan.cu)).

When set for type 1:

- plan construction sets `nf123[idim] = mstu[idim]`: the spread grid is exactly the user
  mode box, with no upsampling
  ([`makeplan.cu:336-343`](../../../src/cuda/makeplan.cu));
- no cuFFT plan and no `fwkerhalf`/fseries kernel are built
  ([`makeplan.cu:175-176`](../../../src/cuda/makeplan.cu),
  [`makeplan.cu:364-398`](../../../src/cuda/makeplan.cu));
- execute allocates no `fw` scratch, uses `fw = fk`, zeroes `blksize*nf`, spreads, and
  `continue`s before FFT and deconvolution
  ([`execute.cu:122-140`](../../../src/cuda/execute.cu));
- the result is the raw spread on the grid of points `-pi + k*(2*pi/nf)` in each dimension,
  not a NUFFT ([`opts.rst:97-121`](../../opts.rst),
  [`c_gpu.rst:386`](../../c_gpu.rst)).

The kernel shape (`ns`, `beta`) is still selected from `tol` and `upsampfac`; the flag
changes the grid size, not the kernel selection.

### 5.2 `modeord`

`modeord` is a deconvolution/output-order option. In `deconv_nd`, it chooses how a user mode
index `k` maps to the fine-grid index `wn` and kernel-half index `fwkerindn`:

- `modeord=0`: `pivot=k-mstu/2`; `wn=pivot` if `>=0`, else `nf+pivot`;
  `fwkerindn=abs(pivot)`.
- `modeord=1`: `pivot=k-mstu+mstu/2`; if `pivot>=0`, `wn=nf+k-mstu`; else `wn=k`;
  `fwkerindn=mstu-k` if `pivot>=0`, else `k`.

See [`execute.cu:19-63`](../../../src/cuda/execute.cu) and the user-facing definition in
[`opts.rst:75-93`](../../opts.rst). Then `fk[fkidx] = fw[fwidx]/kervalue` (type 1) or the
reverse (type 2).

The **spread stage always writes the natural fine-grid order** `fw[ix + nf1*(iy +
nf2*iz)]`. `modeord` never enters `spreadSorted` or any of the spread kernels. In
spread-only mode there is no deconvolution, so `modeord` has no effect at all; the output is
natural fine-grid order. For a full type-1 comparison, both runs must use the same
`modeord`, or apply the documented cyclic shift before comparing. See also the mode-order
note in [`execute.cu:20-22`](../../../src/cuda/execute.cu).

## 6. Numerical comparison oracle

### 6.1 Pinned reference configuration

Use cuFINUFFT on the same GPU/build as the new kernel, and pin every option that can change
the kernel or the order:

```text
type                 = 1
dim                  = 3
nmodes               = (N1, N2, N3)
ntransf              = 1                    // for the first correctness check
tol                  = chosen so cuFINUFFT picks the intended ns
                       (for ns=6, sigma=2: use tol=1e-5)
opts.upsampfac       = 2.0                  // or 1.25; must match the new kernel
opts.gpu_method      = 3                    // OD; do not rely on the auto-choice
opts.gpu_kerevalmeth = same as the new kernel (default 1: Horner)
opts.gpu_spreadinterponly = 1               // raw spread, nf = N, no FFT/deconv
opts.gpu_maxbatchsize = 1                   // with ntransf=1, one batch
opts.gpu_maxsubprobsize = 1024              // pin: changes tile-write grouping/order
opts.gpu_np           = 0                   // pin method-3 staging batch
opts.modeord         = 0                    // irrelevant in spread-only; pin anyway
opts.gpu_binsize{x,y,z} = pinned explicitly       // changes order, not the contribution set
```

Why this pins the semantics:

- `gpu_method=3` is the OD baseline; auto for type 1 is method 2
  ([`makeplan.cu:304-310`](../../../src/cuda/makeplan.cu)).
- `gpu_spreadinterponly=1` gives `nf=N` and removes FFT/deconvolution, so the output is the
  spread grid rather than a NUFFT ([`makeplan.cu:336-353`](../../../src/cuda/makeplan.cu),
  [`execute.cu:122-140`](../../../src/cuda/execute.cu)).
- `gpu_kerevalmeth` chooses direct vs Horner; these are different functions, not different
  summation orders ([`spreadinterp.hpp:91-134`](../../../include/cufinufft/spreadinterp.hpp)).
- `upsampfac`/`tol` fix `ns` and `beta`
  ([`makeplan.cu:83-105`](../../../src/cuda/makeplan.cu)).
- `ntransf=1` avoids batch-stride and repeated-grid complications; test `ntransf>1`
  separately once the single-transform result is correct.

If the new kernel does not go through `cufinufft_plan_t`, replicate these values exactly
(`nf`, `ns`, `beta`, `es_c`, `sigma`, evaluator, folded coordinates, strengths), and
zero the output under the same contract as `execute_type1` before spreading.

### 6.2 Data that must be exercised

Use complex strengths with independent real and imaginary parts and include:

- uniform random points over `[-pi,pi)`;
- points outside `[-pi,pi)` (e.g. `[-3*pi,3*pi)` as in
  [`cufinufft3d_test.cu:42-47`](../../../test/cuda/cufinufft3d_test.cu));
- exact grid-aligned points (all coordinates 0 or integer multiples of `2*pi/nf`);
- points exactly at or immediately around `-pi`, `+pi`, and the periodic seam;
- repeated coordinates;
- clustered points (high per-cell occupancy);
- sparse/empty regions and `M=0`;
- at least one odd-`ns` case (e.g. `ns=5` or `7`) even if the prototype is fixed at
  `ns=6`; this is the only way to observe the unused left halo cell.

The existing 1D spread-only test only checks mass conservation and uses
`gpu_spreadinterponly=1` ([`cufinufft1dspreadinterponly_test.cu:30-38`](../../../test/cuda/cufinufft1dspreadinterponly_test.cu),
[`cufinufft1dspreadinterponly_test.cu:114-117`](../../../test/cuda/cufinufft1dspreadinterponly_test.cu)).
There is no 3D spread-only fine-grid test in the tree, so the prototype needs its own
harness.

### 6.3 Metric and sampling

Copy both fine grids to the host and compare **all** `nf1*nf2*nf3` complex cells:

```text
E_rel = || X_new - R_ref ||_2 / || R_ref ||_2
      = sqrt( sum_cells |X_new - R_ref|^2 / sum_cells |R_ref|^2 )
```

The repo already implements this with the reference as the first argument in
`relerrtwonorm` ([`norms.hpp:21-37`](../../../test/utils/norms.hpp)). Note that this norm
uses `std::norm` (squared magnitude) for both the error and the denominator.

Do **not** sample only the user mode set. In spread-only mode `nf=N`, but boundary cells
(0 and `nf-1` in each dimension) are exactly where halo/wrap bugs appear. Also report:

```text
E_inf = max_cells |X_new - R_ref| / max_cells |R_ref|
```

and, for diagnosis only, the worst per-cell relative error with a floor in the denominator.
A global L2 can hide a small number of bad edge cells if most of the grid is large.

### 6.4 Expected error from a different summation order

If both implementations use the same folded coordinates, the same kernel evaluator and the
same per-point weights, the only differences are:

1. reassociation of the per-cell sum over contributing points;
2. regrouping of the real product `phi_x*phi_y*phi_z` and of `strength*product`.

For a cell receiving `K` contributions of comparable magnitude, statistical rounding gives
a relative difference around

```text
E_rel ~ eps_mach * sqrt(K)
```

(sequential-versus-tree summation differs by a small multiple of this). For `ns=6`:

- `K` is on the order of `ns^3 = 216` for a uniformly occupied grid with `M/nf ~ 1`;
- `K ~ (M/nf)*216` for lower density (and larger for higher density);
- clusters can make `K` much larger.

So FP32 (`eps ~ 1.2e-7`) is expected around `1e-7..1e-6` for moderate occupancy, rising
toward `1e-5` for heavily occupied cells; FP64 (`eps ~ 2.2e-16`) is expected around
`1e-16..1e-14`. These are scaling estimates, not a pass threshold: cancellation can make a
particular cell's relative error larger, and the global L2 weights the large cells.

A difference at the requested NUFFT tolerance (e.g. `1e-4` for FP32) is far too large to
be summation order; it indicates a semantic mismatch such as a wrong fold, start index,
halo, kernel evaluator, or strength pairing.

### 6.5 Calibrate the pass threshold from the reference's own noise

Because the OD path accumulates with shared-memory accumulation followed by global atomics,
the reference itself is not bitwise reproducible. Measure a noise floor on the same problem:

```text
E0 = max(
       relative L2 between two repeated gpu_method=3 spread-only runs,
       relative L2 between gpu_method=3 and gpu_method=1 spread-only runs
     )
```

Both methods use the same `fold_rescale`, the same `get_kerval...` and the same kernel
evaluator; they differ in accumulation order and product grouping. `gpu_method=1` is the
point-driven atomic path ([`spread_nupts_driven_inst.cu:16-62`](../../../src/cuda/spread_nupts_driven_inst.cu));
`gpu_method=3` is the OD path. The existing cross-API test explicitly accepts that atomic
non-determinism prevents bit-exactness
([`cufinufft_simple_test.cu:4-6`](../../../test/cuda/cufinufft_simple_test.cu),
[`cufinufft_simple_test.cu:331-333`](../../../test/cuda/cufinufft_simple_test.cu)).

A practical gate is

```text
E_rel(new, reference) <= max(C * E0, C_abs * eps_mach)
```

with `C` around 4 and a small absolute floor (for example `10*eps_mach`), adjusted only
with evidence from the reference's own repeat/cross-method spread. If `E0` is measured at
FP32 `~2e-7`, the gate is a few `1e-7` to `1e-6`; if the new kernel uses a different
product grouping, expect it to sit nearer the larger end. For FP64, `E0` should be at the
`1e-15` level.

If the new-vs-reference error is much larger than `E0`:

- boundary-only errors: halo size, unused odd-`ns` cell, or one-wrap/modulo bug;
- errors localized at the `-pi`/`+pi` seam: fold or cell-wrap bug;
- errors on grid-aligned points: endpoint evaluator mismatch (direct vs Horner, or CPU
  endpoint convention);
- uniform scaling or sign errors: strength pairing, `idxnupts`, or batch stride;
- random interior errors: missing/duplicated points or wrong kernel support start.

## 7. Contract checklist for the new kernel

A compliant kernel must reproduce:

- [ ] `r = fold_rescale_device(x, nf)` with the cuFINUFFT device formula and rounding;
- [ ] `s = ceil(r - ns/2)` and exactly `ns` samples `u+j`, `j=0..ns-1`;
- [ ] the same kernel evaluator (direct ES or the exact Horner table for the chosen
      `sigma`), not a CPU PSWF or an "equivalent" analytic formula;
- [ ] the same `ns`, `beta`, `es_c`, `sigma` chosen by the pinned plan configuration;
- [ ] complex interleaved strengths `c[j]`, paired with the same original point index;
- [ ] every point whose support covers a cell contributes to that cell, regardless of which
      bin owns the point or the cell, with no dropped or duplicated tap;
- [ ] per-transform strides `c + t*M`, `fw + t*nf`, x-fastest flat fine grid;
- [ ] periodic cell wrap for every support cell, including `s<0`, `s+ns-1>=nf` and
      `r=nf` from fold roundoff;
- [ ] an output that is zero where the reference is zero, with either a full-grid write or
      the same pre-zero contract as `execute_type1`;
- [ ] the raw spread grid in natural order; `modeord` must not be applied in the spread
      stage;
- [ ] the same result for even and odd `ns`, including the odd-`ns` unused left halo cell.

## 8. Source map

| Topic | Source |
|---|---|
| CPU fold/rescale | [`include/finufft/simd.hpp:321-328`](../../../include/finufft/simd.hpp) |
| CUDA fold/rescale and directed rounding | [`include/cufinufft/spreadinterp.hpp:57-89`](../../../include/cufinufft/spreadinterp.hpp) |
| CUDA start index and kernel samples | [`include/cufinufft/spreadinterp.hpp:154-196`](../../../include/cufinufft/spreadinterp.hpp) |
| CUDA direct ES evaluator | [`include/cufinufft/spreadinterp.hpp:91-111`](../../../include/cufinufft/spreadinterp.hpp) |
| CUDA Horner evaluator / tables | [`include/cufinufft/spreadinterp.hpp:113-134`](../../../include/cufinufft/spreadinterp.hpp) |
| CUDA bin layout, padded tile, local→global wrap | [`include/cufinufft/spreadinterp.hpp:136-262`](../../../include/cufinufft/spreadinterp.hpp) |
| CUDA bin histogram and inverse permutation | [`include/cufinufft/spreadinterp.hpp:300-325`](../../../include/cufinufft/spreadinterp.hpp) |
| OD kernel | [`src/cuda/spread_output_driven_inst.cu:16-128`](../../../src/cuda/spread_output_driven_inst.cu) |
| OD preprocessing | [`src/cuda/spread_subprob_inst.cu:136-181`](../../../src/cuda/spread_subprob_inst.cu) |
| GPU plan data (`es_c`, `es_beta`, `idxnupts`) | [`include/cufinufft/cufinufft_plan_t.hpp:530-598`](../../../include/cufinufft/cufinufft_plan_t.hpp) |
| GPU `ns`/`beta`/`sigma` setup | [`src/cuda/makeplan.cu:34-108`](../../../src/cuda/makeplan.cu) |
| GPU auto sigma and nf sizing | [`src/cuda/makeplan.cu:283-353`](../../../src/cuda/makeplan.cu), [`src/cuda/fft.cu:166-179`](../../../src/cuda/fft.cu) |
| GPU `setpts` stores raw pointers | [`src/cuda/setpts.cu:27-77`](../../../src/cuda/setpts.cu) |
| GPU execute, batch strides, zeroing, spread-only | [`src/cuda/execute.cu:105-149`](../../../src/cuda/execute.cu) |
| `modeord` deconvolution mapping | [`src/cuda/execute.cu:19-63`](../../../src/cuda/execute.cu) |
| CPU subgrid, padding, wrapped merge | [`include/finufft/spread.hpp:386-451`](../../../include/finufft/spread.hpp), [`include/finufft/spread.hpp:791-868`](../../../include/finufft/spread.hpp) |
| CPU 3D spread kernel | [`include/finufft/spread.hpp:303-384`](../../../include/finufft/spread.hpp) |
| CPU PSWF kernel / Horner setup | [`src/common/kernel.cpp:19-138`](../../../src/common/kernel.cpp), [`include/finufft/makeplan.hpp:203-300`](../../../include/finufft/makeplan.hpp) |
| Relative L2 helper | [`test/utils/norms.hpp:21-37`](../../../test/utils/norms.hpp) |
| Spread-only / modeord docs | [`docs/opts.rst:75-121`](../../opts.rst), [`docs/c_gpu.rst:380-423`](../../c_gpu.rst) |

## 9. Explicit non-answers

- This note does not choose a tile geometry, tile size, Morton key, or ownership rule for
  the new kernel; those are design/prototype decisions for later tickets.
- This note does not claim bitwise equality with any existing path. Atomic ordering,
  `idxnupts` tie order, and `gpu_binsize` can change the summation order without changing
  the mathematical spread.
- This note does not treat a CPU FINUFFT spread or a full NUFFT output as the oracle for
  the GPU kernel. Both add transformations that are outside the spread contract.
