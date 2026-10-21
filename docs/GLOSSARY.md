# FINUFFT

FINUFFT computes the three standard types of nonuniform fast Fourier transform (NUFFT)
in one, two, or three dimensions to a requested relative tolerance, on CPU or GPU.

## Language

### Transforms

**NUFFT**:
A fast approximation to a Fourier sum evaluated at nonuniformly spaced points, computed in
near-linear time to a requested relative accuracy rather than by direct summation.
_Avoid_: NFFT (a different library), nonuniform DFT, direct sum

**type 1**:
The nonuniform-to-uniform transform: given strengths $c_j$ at nonuniform points $x_j$,
evaluate Fourier mode coefficients $f_k = \sum_j c_j e^{\pm i k \cdot x_j}$.
_Avoid_: inverse NUFFT, inverse of type 2, forward transform (use "type 1")

**type 2**:
The uniform-to-nonuniform transform: given Fourier mode coefficients $f_k$, evaluate
$c_j = \sum_k f_k e^{\pm i k \cdot x_j}$ at nonuniform points $x_j$.
_Avoid_: inverse NUFFT, inverse of type 1, forward transform (the docs use "forward"
for type 2, but "type 2" is unambiguous)

**type 3**:
The nonuniform-to-nonuniform transform: given strengths $c_j$ at nonuniform points $x_j$
and nonuniform frequency targets $s_k$, evaluate
$f_k = \sum_j c_j e^{\pm i s_k \cdot x_j}$.
_Avoid_: type 1 with nonuniform frequencies, type 2 with nonuniform frequencies

**adjoint transform**:
The transform obtained by swapping input and output roles and reversing the exponential
sign; type 2 is the adjoint of type 1 with the opposite sign, and type 3 is its own
adjoint with sources and frequency targets swapped.
_Avoid_: inverse (the NUFFT types are adjoints, not inverses), transpose

**mode**:
An integer frequency index $k$ in the product set $K_N$; the number of modes in dimension
$i$ is $N_i$.
_Avoid_: frequency (reserve that for type 3 targets), grid point, bin

**mode coefficient**:
The complex value $f_k$ associated with one Fourier mode; an output for type 1 and an
input for type 2.
_Avoid_: strength, amplitude, sample; using it for type 3's values at nonuniform
frequency targets

**mode ordering**:
The order in which Fourier mode indices are stored: CMCL ordering or FFT ordering.
_Avoid_: layout, stride, indexing scheme

**CMCL ordering**:
The default mode order (`modeord=0`): indices increase from $-N/2$ to $N/2-1$ for even
$N$, and from $-(N-1)/2$ to $(N-1)/2$ for odd $N$.
_Avoid_: increasing order, standard order

**FFT ordering**:
The mode order selected by `modeord=1`: indices start at zero and wrap to negative indices
halfway through the array, so the two orders differ by an `fftshift`.
_Avoid_: fftshift order (fftshift is the operation, not the name), wraparound order

### Nonuniform data

**nonuniform point**:
A real-valued location $x_j$ at which a transform is evaluated or from which it spreads;
types 1 and 2 treat points as periodic modulo $2\pi$, while type 3 points have no
periodicity.
_Avoid_: node, sample, scatter point, off-grid point

**source**:
A nonuniform point that carries an input strength: the spatial points of types 1 and 3.
_Avoid_: input point, source node, sample

**target**:
A point at which output values are requested: the nonuniform points of type 2 and the
nonuniform frequency targets of type 3.
_Avoid_: output point, sink; destination is reserved for wayfinder maps

**strength**:
The complex coefficient $c_j$ attached to a source in type 1 or type 3.
_Avoid_: weight (reserve that for kernel weights), amplitude, coefficient

**frequency target**:
A nonuniform frequency $s_k$ at which a type 3 transform is evaluated; distinct from an
integer Fourier mode.
_Avoid_: mode, frequency bin, k-point

**space-bandwidth product**:
The product of the source-domain width and the target-frequency-domain width in each
dimension; it controls the cost and internal grid size of a type 3 transform.
_Avoid_: problem size, $N \times M$

### Algorithm and accuracy

**spreading**:
Depositing source strengths onto nearby points of the internal fine grid using the
spreading kernel; it is part of types 1 and 3.
_Avoid_: gridding, scatter, deposit

**interpolation**:
Sampling the internal fine grid at nonuniform target points using the spreading kernel; it
is part of types 2 and 3.
_Avoid_: gridding, gather, resampling

**fine grid**:
The internal uniform grid on which the FFT is performed; its per-dimension size is derived
from the mode count, the upsampling factor, the kernel width, and a supported FFT size.
_Avoid_: oversampled grid, padded grid, mode grid

**upsampling factor**:
The ratio $\sigma$ between the fine-grid size and the requested mode count per dimension;
it trades kernel width, accuracy, FFT cost, and memory.
_Avoid_: oversampling ratio, padding factor

**spreading kernel**:
The compactly supported function that assigns fine-grid weights during spreading and
interpolation; its width and shape set the accuracy/cost trade-off.
_Avoid_: gridding kernel, window function, filter

**kernel width**:
The number of fine-grid points spanned by the spreading kernel in each dimension, denoted
$n_s$; a larger width allows higher accuracy at higher cost.
_Avoid_: support size, stencil width

**PSWF kernel**:
The prolate spheroidal wavefunction kernel used by the CPU library, chosen for near-optimal
aliasing error at a given kernel width.
_Avoid_: ES kernel, Kaiser-Bessel kernel, window

**ES kernel**:
The exponential-of-semicircle kernel still used by cuFINUFFT at upsampling factors 2.0 and
1.25; the CPU library used it before v2.6.
_Avoid_: PSWF kernel, Kaiser-Bessel kernel, legacy kernel (be explicit about the version)

**deconvolution**:
The stage that removes the spreading kernel's effect by dividing by the kernel's Fourier
transform; in types 1 and 2 it also reorders modes.
_Avoid_: inverse filtering, amplitude correction, unspreading

**kernel Fourier transform**:
The Fourier transform of the spreading kernel, evaluated on the fine grid or mode grid and
used by deconvolution.
_Avoid_: kernel FT, window transform

**bin sort**:
A spatial reordering of nonuniform points, stored as a permutation, that improves cache
locality during spreading and interpolation.
_Avoid_: sort, reorder, bucketing

**requested tolerance**:
The relative accuracy requested through `tol` or `eps`; FINUFFT uses it to choose kernel
and grid parameters, but it is a target rather than a guarantee.
_Avoid_: error, epsilon (reserve that for machine epsilon), precision

### Interfaces and execution

**simple interface**:
The one-call API that prepares and executes a single transform.
_Avoid_: basic API, direct interface

**vectorized interface**:
The "many" API that performs `ntr` transforms with the same nonuniform points but separate
stacked strength or mode vectors.
_Avoid_: batch interface, multi-transform API

**guru interface**:
The plan-based API with separate `makeplan`, `setpts`, `execute` (or `execute_adjoint`),
and `destroy` calls, allowing reuse and forward-adjoint pairs.
_Avoid_: advanced interface, low-level API

**plan**:
The opaque object that holds a transform's type, dimension, mode sizes, options, and
internal kernel and FFT setup; it can be executed repeatedly.
_Avoid_: handle, context, workspace

**setpts**:
The plan call that supplies nonuniform point arrays; the plan retains pointers to them, so
the arrays must not change before execution.
_Avoid_: load points, bind points

**execute**:
The plan call that performs the planned transform; `execute_adjoint` performs its adjoint
instead.
_Avoid_: run, apply, compute

**transform stack**:
The `ntr` transforms sharing a plan and nonuniform points but using separate strength or
mode vectors; the transform index is the slowest-varying array dimension.
_Avoid_: batch, bundle, vector

**batch**:
A group of stacked transforms processed together in one FFT during execution; it is an
internal scheduling unit, not the user's transform stack.
_Avoid_: using batch for `ntr` or for the transform stack

### Implementations

**FINUFFT**:
The project's CPU library, written in C++ with OpenMP and SIMD, using FFTW or DUCC0 for the
fine-grid FFT.
_Avoid_: CPU NUFFT (too generic)

**cuFINUFFT**:
The CUDA GPU implementation integrated in the FINUFFT repository.
_Avoid_: GPU FINUFFT, CUDA NUFFT

**`finufft` / `finufftf`**:
The C/C++ entry-point prefixes for double and single precision, respectively; the trailing
`f` selects single precision.
_Avoid_: using `finufft` for both precisions, float API

**FFT backend**:
The FFT implementation used for the fine-grid transform, selected at build time: FFTW or
DUCC0.
_Avoid_: FFT library, FFT engine

**spread/interp-only mode**:
The expert option that exposes only spreading (type 1) or interpolation (type 2), omitting
the FFT and deconvolution stages.
_Avoid_: gridding-only mode, no-FFT mode
