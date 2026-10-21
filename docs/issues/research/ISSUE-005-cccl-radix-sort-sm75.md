# ISSUE-005 research: CCCL/CUB radix sort for Morton keys on sm_75

**Date:** 2026-10-21
**Ticket:** [ISSUE-005](../done/ISSUE-005-cccl-radix-sort-sm75.md)
**Design context:** [destination-owned NUFFT sketch](../assets/ISSUE-001-destination-owned-nufft-design.md)
**Machine:** Quadro T1000 (sm_75, 4 GB), CUDA 12.8 toolkit at `~/cuda-12.8` (`nvcc` 12.8.61)

## TL;DR

- The build requests CCCL major 3. The toolkit under `/home/paseo/cuda-12.8` ships
  CCCL **2.7.0** (CUB `200700`), whose config is incompatible with a major-3 request, so
  CMake/CPM will fetch **CCCL v3.4.2** (CUB `300402`) from GitHub. No fetched tree exists
  in this worktree yet because no GPU build has been configured here.
- CUB 3.4.2 is header-only and exposes the legacy
  `cub::DeviceRadixSort::SortPairs<KeyT, ValueT>` API, a `cub::DoubleBuffer` variant, and
  a newer env-based API (3.4.0+). For `(uint64_t, uint32_t)` on sm_75, the legacy
  `DoubleBuffer` API is the simplest robust choice.
- 64-bit keys are directly supported; no decomposer is needed. On sm_75, CUB uses its
  sm_70 tuning policy and the 8-bit "onesweep" path, so a full 64-bit key costs **8 radix
  passes**; a 32-bit key costs 4. On this T1000 a full 64-bit pair sort took about **2x**
  the time of the same sort restricted to the used Morton bits.
- The biggest, simplest optimization is `end_bit = B`, where `B` is the number of
  significant Morton bits actually used. On the measured T1000 this was ~2x faster than
  sorting all 64 bits. A 32-bit key is only slightly faster than a 64-bit key once the
  pass count is equal, and is usually available on a 4 GB T1000 for a grid that fits, but
  `B` must be checked including ghost/output-bin padding.
- `DeviceRadixSort` is **stable**, so equal Morton keys preserve input point order; this
  makes run construction deterministic for a fixed input ordering. A 4-byte payload is
  well within CUB's supported payload usage.
- Temporary-storage sizing uses the standard CUB `nullptr` query. It is host-side only:
  no kernel is launched and no stream synchronization is performed. The pointer API
  allocates its own `N`-sized alternate key/value buffers in temp storage; the
  `DoubleBuffer` API avoids those because the caller supplies the alternate buffers. CUB
  docs describe the `DoubleBuffer` temp as `O(P)`, but the onesweep lookback array still
  grows with `N / items_per_thread`; measured temp was still small (about 0.37 B/item at
  N = 1M).

## 1. Which CCCL/CUB is actually used

### 1.1 CMake resolution

The top-level CMake sets:

- `CUDA12_CCCL_VERSION "3.4.2"` at `CMakeLists.txt:84`.
- `cmake/setupCCCL.cmake:19-36` uses
  `cpmfindpackage(NAME CCCL VERSION 3 ... GIT_TAG v${CUDA12_CCCL_VERSION})`.

Two details matter:

1. The version floor is **major 3**, not exactly 3.4.2. If a system CCCL 3.0/3.1/3.2 were
   installed, `find_package` would accept it and the 3.4-only env API might be missing.
   On this machine no such system CCCL exists, so CPM falls through to the Git tag
   `v3.4.2`.
2. `CUDA12_CCCL_VERSION` only controls the **fetched** tag; it does not pin an installed
   CCCL.

The `SYSTEM NO` flag on the fetch is also important. It makes CPM add the fetched include
directories as `-I` rather than `-isystem`; `nvcc` puts its own toolkit include directory
after every `-I` but before `-isystem` paths, so this ordering is what lets the fetched
3.4.2 `cub/` and `cuda/std/` headers win over the toolkit's 2.7 headers. See the
CUDA-11 branch comment at `cmake/setupCCCL.cmake:4-8` and the CCCL note at
`/tmp/cccl-3.4.2/README.md:138-139`.

The toolkit's installed CCCL is 2.7.0:

- `/home/paseo/cuda-12.8/include/cub/version.cuh:61` — `#define CUB_VERSION 200700`.
- `/home/paseo/cuda-12.8/lib64/cmake/cccl/cccl-config-version.cmake:1-17` — major 2,
  minor 7; `PACKAGE_VERSION_COMPATIBLE` is false for a major-3 request.
- `/home/paseo/cuda-12.8/version.json` — `cuda_cccl` version `12.8.55`.

Therefore the CUDA 12.8 build fetches CCCL 3.4.2 rather than using the toolkit's CUB 2.7.
A minimal `find_package(CCCL 3 CONFIG PATHS <toolkit cmake dirs>)` probe with CUDA
language enabled returned `CCCL_FOUND=0`, confirming that the toolkit's 2.7 package is
rejected for a major-3 request. The expected fetched location with the `gpu` preset is
`build/gpu/_deps/cccl-src/` (CPM/FetchContent default). It is absent here:

- `build/gpu` does not exist;
- `build/dev/CMakeCache.txt` and `build/default/CMakeCache.txt` both have
  `FINUFFT_USE_CUDA:BOOL=OFF`.

To inspect the exact headers without touching the repo build, I cloned tag `v3.4.2` to
`/tmp/cccl-3.4.2` (HEAD `d36012203ef73ac7f966e848dd88482273e91e02`, tag `v3.4.2`).
All CCCL 3.4.2 citations below are from that tree. When the GPU build is configured, the
same files live under `build/gpu/_deps/cccl-src/`.

### 1.2 Version and support facts

- `CUB_VERSION` is `300402`:
  `/tmp/cccl-3.4.2/cub/cub/version.cuh:37`.
- `CCCL_VERSION` in the CMake package is 3.4.2 by construction from tag `v3.4.2`.
- CCCL 3.4.2 is header-only:
  `/tmp/cccl-3.4.2/README.md:112-139`.
- CCCL 3.x supports CUDA 12.x/13.x latest patch releases:
  `/tmp/cccl-3.4.2/README.md:241-247`. CUDA 12.8.61 is in range.
- CCCL's CMake architecture floor is sm_75:
  `/tmp/cccl-3.4.2/cmake/CCCLCheckCudaArchitectures.cmake:17-18`
  (`set(minimum_cccl_arch 75) # 13.x dropped below Turing`). sm_75 is supported.
- FINUFFT's CUDA target is C++17:
  `src/cuda/CMakeLists.txt:101-109`; CUB 3.4.2 headers compile under C++17 with CUDA
  12.8 (probe below).
- `CCCL::CCCL` is an interface target and CCCL is header-only, so there is no CCCL
  library to link: `src/cuda/CMakeLists.txt:119-129`.
- The env API's doc note says "First appears in CUDA Toolkit 13.4"
  (`device_radix_sort.cuh:377-378`), but the fetched 3.4.2 headers compile and run with
  CTK 12.8 in the probe below; the env API is simply not present in the toolkit's CUB
  2.7 headers.

## 2. Available CUB APIs, headers, and namespaces

All device radix-sort entry points are in namespace `cub`, declared in
`<cub/device/device_radix_sort.cuh>` (also pulled in by `<cub/cub.cuh>`). The
double-buffer wrapper is `cub::DoubleBuffer<T>` in `<cub/util_type.cuh>`.

| API | Header | Key points |
|---|---|---|
| `cub::DeviceRadixSort::SortPairs` (pointer API) | `<cub/device/device_radix_sort.cuh>` | Separate contiguous key and value arrays; input is `const`; no overlap between any input/output ranges; CUB supplies internal alternate buffers in temp storage; `begin_bit`/`end_bit`; explicit `cudaStream_t`. Signature at `device_radix_sort.cuh:334-345`. |
| `cub::DeviceRadixSort::SortPairs` (`DoubleBuffer`) | same | Caller supplies two key buffers and two value buffers; `is_overwrite_okay=true`; both buffers may be altered; `d_keys.Current()` / `d_values.Current()` point at the result after the call; temp storage excludes the caller's `N`-sized buffers (measured ~0.37 B/item at N = 1M). Signature at `device_radix_sort.cuh:1041-1057`; docs at `:1086-1088` and `:249-250`. |
| `cub::DeviceRadixSort::SortPairs` (env API) | same | New in 3.4.0; no explicit temp storage; stream from `cuda::get_stream`, allocator from `cuda::mr::get_memory_resource`. Signature at `device_radix_sort.cuh:450-463`; version note at `:374-378`. |
| `cub::DeviceRadixSort::SortPairsDescending` | same | Descending variant; same pointer/DoubleBuffer/env shapes. |
| `cub::DeviceRadixSort::SortKeys` / `SortKeysDescending` | same | Keys-only; use if the payload can be discarded or recomputed. |
| `cub::DeviceSegmentedRadixSort::SortPairs` | `<cub/device/device_segmented_radix_sort.cuh>` | Independent segments; not needed for one Morton array, but available. |
| `cub::DeviceMergeSort::SortPairs` | `<cub/device/device_merge_sort.cuh>` | Comparison sort; **not stable**; CUB documents it as slower than `DeviceRadixSort` for arithmetic keys (`device_merge_sort.cuh:31-36`). |
| `cub::DeviceMergeSort::StableSortPairs` | same | Stable merge sort; still slower than radix for arithmetic keys. |
| `cub::DeviceSegmentedSort::StableSortPairs` | `<cub/device/device_segmented_sort.cuh>` | Comparison-based segmented stable sort; not a speed alternative. |

There is no separate `cub::DeviceSort` device-wide wrapper in CCCL 3.4.2; use the radix or
merge-sort APIs above.

Header-only / linking: include the header; `CCCL::CCCL` is an `INTERFACE` imported target
and no CCCL `.so` is involved. CUDA runtime is still required and is already linked as
`CUDA::cudart`.

## 3. 64-bit keys and sm_75/Turing behavior

### 3.1 Supported key/value types

- `DeviceRadixSort` supports all built-in C++ numeric primitive types, including 64-bit
  unsigned integers. `uint64_t` is used directly as `KeyT`; no decomposer is required:
  `device_radix_sort.cuh:80-86`.
- For unsigned integral keys, CUB uses the key bits directly; no sign/float twiddle
  transformation is applied: `device_radix_sort.cuh:113-119`.
- User-defined key types require a decomposer returning a tuple of references to
  arithmetic fields (`device_radix_sort.cuh:162-170` and `:193-201`). A plain
  `uint64_t` Morton key does not need one.
- `ValueT` (the payload) is carried along, not radix-decomposed. CUB's own pair tests
  instantiate `ValueT` as `uint8_t`, `uint64_t`, and a user-defined
  `custom_type_t` (`cub/test/catch2_test_device_radix_sort_pairs.cu:28-29`); the custom
  type has two `size_t` members plus policy bytes (`c2h/include/c2h/custom_type.h:13-17`),
  i.e. larger than 8 bytes. A 4-byte payload is trivial for CUB.

### 3.2 Which tuning policy sm_75 gets

CUB 3.4.2 has policy specializations `Policy500`, `600`, `610`, `620`, `700`, `800`,
`900`, `1000`; there is **no `Policy750`**. The selector uses the `cc >= 7.0` branch for
all sm_7x, so sm_75 uses the Volta-era `Policy700` tuning:

- `tuning_radix_sort.cuh:1332-1356` — `Policy700` constants and `ONESWEEP =
  sizeof(KeyT) >= sizeof(uint32_t)`, `ONESWEEP_RADIX_BITS = 8`, and
  `AgentRadixSortOnesweepPolicy<256, ...>`.
- `tuning_radix_sort.cuh:1889-1909` — the runtime selector branch for `cc >= 7.0`;
  comments cite GV100/V100, not Turing. The onesweep policy uses
  `RADIX_RANK_MATCH_EARLY_COUNTS_ANY` (which relies on `match.any.sync`, available on
  sm_75) and `RADIX_SORT_STORE_DIRECT`.
- `tuning_radix_sort.cuh:1350` and `:1903` — onesweep items per thread is `46` for
  `4-byte key + 4-byte value`, but `23` for an 8-byte key (including `uint64_t` with a
  4-byte payload). So a 64-bit key reduces per-thread work and increases relative
  overhead. CUB's dominant type is `max(sizeof(KeyT), sizeof(ValueT))`
  (`tuning_radix_sort.cuh:973`); with a 4-byte payload the 8-byte key dominates.

The dispatch chooses onesweep for key sizes >= 4 bytes
(`dispatch_radix_sort.cuh:1068-1073`, `tuning_radix_sort.cuh:1893`). Consequently:

- 64-bit key: `ceil(64/8) = 8` radix passes.
- 32-bit key: `ceil(32/8) = 4` radix passes.
- `end_bit = 30`: `ceil(30/8) = 4` passes even with a `uint64_t` key.

For very small inputs the single-tile kernel is used instead of onesweep when
`num_items <= threads_per_block * items_per_thread`; for `Policy700` that is
`256 * 19 = 4864` items (`dispatch_radix_sort.cuh:1060-1063`,
`tuning_radix_sort.cuh:1393-1402`). Typical FINUFFT point counts are far above this, but
small-M behavior should be measured separately.

### 3.3 Measured behavior on the T1000

I compiled a small probe in `/tmp` with the repo's CUDA flags and the fetched headers:

```bash
/home/paseo/cuda-12.8/bin/nvcc -std=c++17 -arch=sm_75 \
  --extended-lambda --extra-device-vectorization \
  -I/tmp/cccl-3.4.2/cub \
  -I/tmp/cccl-3.4.2/libcudacxx/include \
  -I/tmp/cccl-3.4.2/thrust \
  /tmp/issue005_bench.cu -o /tmp/issue005_bench
```

The benchmark sorts random `(uint64_t key, uint32_t value)` pairs with the `DoubleBuffer`
API, preallocated temp storage, and a dedicated CUDA stream. It reports the minimum of 10
post-warmup CUDA-event timings; it is indicative, not a substitute for the prototype's
end-to-end measurement. Run-to-run clock variation was noticeable on this small GPU (one
4M full-64-bit median was 14.7 ms), so the table uses the minimum sample.

| key type | `end_bit` | passes | n = 1,048,576 | n = 4,194,304 |
|---|---:|---:|---:|---:|
| `uint64_t` | 64 | 8 | 3.39 ms | 9.40 ms |
| `uint64_t` | 32 | 4 | 1.71 ms | 4.85 ms |
| `uint64_t` | 30 | 4 | 1.68 ms | 4.81 ms |
| `uint32_t` | 32 | 4 | 1.57 ms | 4.91 ms |
| `uint32_t` | 30 | 4 | 1.56 ms | 4.89 ms |

Interpretation:

- Restricting the radix range to the bits actually used is the dominant win: full 64-bit
  was ~2.0x slower than `end_bit = 30`.
- Once the pass count is equal, 32-bit vs 64-bit key width was only ~7% at 1M and within
  noise at 4M. The key width itself is not the main problem; the number of passes is.
- CUB's sm_70 policy is tuned on GV100/V100, so absolute throughput on the T1000
  should not be extrapolated to other Turing parts without measurement.

The exact `DoubleBuffer` temp-storage query for 64-bit key + 32-bit value was
`390,655` bytes at n = 1,048,576 (0.37 B/item); the pointer API query was
`12,973,567` bytes (12.37 B/item), exactly `12 * n + 390,655`. The 32-bit-key
`DoubleBuffer` query was about 0.09 B/item.

## 4. Temporary storage, in-place behavior, and streams

### 4.1 Pointer API versus `DoubleBuffer`

The pointer API is not in-place and does not overlap buffers:

- docs: `device_radix_sort.cuh:236-245` ("contents of the input data are not altered",
  "in-place operations are not supported", no overlap between key/value input/output
  ranges);
- source: `device_radix_sort.cuh:354-360` (`is_overwrite_okay = false`, internal
  `DoubleBuffer` around caller pointers);
- temp storage formula for the onesweep path:
  `dispatch_radix_sort.cuh:562-573`:
  - bins (`num_portions * num_passes * radix_digits * sizeof(OffsetT)`),
  - lookback (`max_num_blocks * radix_digits * sizeof(AtomicOffsetT)`),
  - extra key buffer (`num_items * sizeof(KeyT)`) and extra value buffer
    (`num_items * sizeof(ValueT)`) when `is_overwrite_okay` is false and there is more
    than one pass,
  - counters.

The `DoubleBuffer` API sets `is_overwrite_okay = true`
(`device_radix_sort.cuh:1057`), so those two `num_items * sizeof(...)` allocations
disappear. The caller supplies the alternate buffers:

- docs: `device_radix_sort.cuh:939-941` and `:1086-1088` — both buffers in each pair may
  be altered; in-place/overlap is still not supported;
- docs: `device_radix_sort.cuh:249-250` — CUB describes the `DoubleBuffer` path as using
  only `O(P)` temporary storage. In the source, the remaining temp is bins, the lookback
  array (`max_num_blocks * radix_digits * sizeof(AtomicOffsetT)`), and counters
  (`dispatch_radix_sort.cuh:562-573`). `max_num_blocks` scales with
  `num_items / onesweep_tile_items`, so the lookback part still grows with `N`; it is
  simply much smaller than the pointer API's `N`-sized key/value buffers;
- `DoubleBuffer` definition and `Current()` / `Alternate()`:
  `cub/util_type.cuh:745-774`;
- after the call CUB updates the selector to the final output
  (`device_radix_sort.cuh:1017-1025`).

For a `(uint64_t, uint32_t)` sort at n = 1M, the memory comparison is:

- pointer API: input `12n` + output `12n` + temp `(12n + 0.37n)` ≈ `36n`;
- `DoubleBuffer`: current `12n` + alternate `12n` + temp `0.37n` ≈ `24n`.

So `DoubleBuffer` both reduces temp storage and lets the plan reuse its own buffers. The
`Current()` pointers must be used after the call; do not assume the result is always in
buffer 0.

### 4.2 Querying temp storage without a host synchronization

The standard CUB convention is:

```cpp
void* d_temp = nullptr;
size_t temp_bytes = 0;
cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, ...);
// temp_bytes now holds the required size; no work was done
cudaMalloc(&d_temp, temp_bytes);   // or an async/plan allocator
cub::DeviceRadixSort::SortPairs(d_temp, temp_bytes, ...); // execute
```

Primary source:

- `device_radix_sort.cuh:299-302` — when `d_temp_storage == nullptr`, the required size
  is written and no work is done.
- `dispatch_radix_sort.cuh:582-586` — the onesweep path returns immediately after
  computing the allocation sizes when temp storage is null.
- `dispatch_radix_sort.cuh:1035-1043` — empty input (or no bits to sort with overwrite
  allowed) returns `temp_storage_bytes = 1`; no kernel launch.

The query is host-side. It can involve `cudaGetDevice` / `cudaDeviceGetAttribute` for
device/policy selection, but it does **not** launch a kernel and does **not** call
`cudaStreamSynchronize` or `cudaDeviceSynchronize`. The actual sort is asynchronous on the
supplied stream; error checking should be done with the project's usual
`THROW_IF_CUDA_ERROR()` / `cudaGetLastError()` pattern, not by synchronizing.

The size depends on:

- `num_items`,
- `sizeof(KeyT)` and `sizeof(ValueT)`,
- `begin_bit` / `end_bit` (pass count),
- pointer API vs `DoubleBuffer` (`is_overwrite_okay`),
- the selected device/tuning policy.

It is therefore a per-plan value, not a compile-time constant. Query once for the maximum
expected `M`/`B` and reuse the allocation for repeated transforms with the same shape, or
re-query when the shape changes. In particular, the pointer API's temp storage includes
`N * (sizeof(KeyT) + sizeof(ValueT))`, so sizing it for a smaller `M` is not enough when
`M` grows. Query on the same CUDA device that will execute the sort: CUB selects the
tuning policy from the current device's compute capability.

### 4.3 Stream and allocator integration

- The legacy APIs take an explicit `cudaStream_t` (last argument, default `nullptr`). All
  CUB kernels and internal `cudaMemsetAsync` calls are ordered on that stream. Pass
  `d_plan.stream`; do not rely on the legacy default stream.
- For temp storage, `ThrustAllocatorAsync<T>` is already the plan's stream/device-aware
  allocator: `include/cufinufft/cufinufft_plan_t.hpp:140-181`. It uses
  `cudaMallocAsync`/`cudaFreeAsync` when `cudaDevAttrMemoryPoolsSupported` is set, and
  falls back to `cudaMalloc`/`cudaFree` otherwise. `gpu_scratch<T>` wraps it as RAII
  scratch (`:185-203`). Use the same stream as the CUB call so the stream-ordered free is
  correctly ordered after the sort.
- On this T1000, `cudaDevAttrMemoryPoolsSupported` is `1`, so the async path is active.
  FINUFFT already records this in `GpuCapabilities::memory_pools_supported`
  (`include/cufinufft/types.hpp:61-85`) and warns when a stream is used without pool
  support (`src/cuda/makeplan.cu:261-265`).
- The new env API is optional. Its default memory resource is
  `cub::detail::device_memory_resource`, which calls `cudaMallocAsync`/`cudaFreeAsync` on
  the queried stream (`cub/detail/device_memory_resource.cuh:31-94`). The env dispatch
  queries the size, allocates, executes, and deallocates on every call
  (`cub/detail/env_dispatch.cuh:40-75`). That is convenient but adds an allocation/free
  per sort; for repeated sorts the legacy API with a cached temp buffer is better.
- A custom env memory resource must satisfy `cuda::mr::resource`: both
  `allocate_sync`/`deallocate_sync` and stream-ordered
  `allocate(stream_ref, ...)`/`deallocate(stream_ref, ...)`
  (`libcudacxx/include/cuda/__memory_resource/resource.h:53-78`). `ThrustAllocatorAsync`
  is not itself a `cuda::mr::resource`, so using it with the env API would need a small
  adapter. The legacy API avoids that work.
- The env API's default resource assumes `cudaMallocAsync` works; it does not consult
  `cudaDevAttrMemoryPoolsSupported`. On a device without pools, the plan allocator's
  fallback is safer. This is another reason to use the legacy API for the first
  prototype.

## 5. Determinism, stability, and payload limits

### 5.1 Stability and determinism

CUB documents `DeviceRadixSort` as **stable**:

- `device_radix_sort.cuh:131-136` — "DeviceRadixSort is stable. For floating-point types,
  -0.0 and +0.0 are considered equal and appear in the result in the same order as they
  appear in the input."

A small sm_75 probe with duplicate keys produced:

```text
input keys:    5 5 3 5 3 9 3 5
input values:  0 1 2 3 4 5 6 7
output keys:   3 3 3 5 5 5 5 9
output values: 2 4 6 0 1 3 7 5
```

The payloads for equal keys are in input order (`2,4,6` for key 3; `0,1,3,7` for key 5),
as stability requires. For a fixed input ordering the sorted output is therefore
deterministic, including the payload order within equal Morton keys. This is stronger than
"deterministic" alone: it is exactly the property run construction needs, because equal
Morton keys remain contiguous and their payloads can be processed in a fixed order.

If a canonical payload order within a run is ever required, sort by `(Morton, point index)`
rather than Morton alone (e.g. pack the index into a wider key or use a decomposer). The
prototype's run construction only needs grouping, so this is not necessary initially.

### 5.2 Payload limits

- There is no documented API-level maximum for `ValueT`; it is carried as an opaque
  payload. CUB's tests exercise `uint8_t`, `uint64_t`, and a user-defined value type
  larger than 8 bytes (`cub/test/catch2_test_device_radix_sort_pairs.cu:28-29`;
  `c2h/include/c2h/custom_type.h:13-17`). A 16-byte `Payload16` probe also compiled and
  queried temp storage successfully.
- Practical limits are register/shared-memory pressure and the temp-storage extra value
  buffer for the pointer API (`num_items * sizeof(ValueT)`). A 4-byte payload is
  negligible; in the `DoubleBuffer` path the caller's alternate value buffer is `4n`
  bytes.
- `ValueT` should be safe to move/assign bytewise in device code (trivially copyable in
  practice); `uint32_t` is.
- The point index as `uint32_t` covers `M <= 2^32-1`. FINUFFT's `M` is currently `int`
  (`CUFINUFFT_BIGINT = int`, `include/cufinufft/types.hpp:17`), so a `uint32_t` payload is
  safe. If `M` ever exceeds 2^32, switch to a 64-bit payload; CUB supports it.
- For `num_items`, pass an `int` while `M` fits. CUB's `choose_offset_t` selects a
  32-bit unsigned offset for integral types up to 4 bytes
  (`cub/detail/choose_offset.cuh:34-43`), which is cheaper than 64-bit offsets. Passing
  `size_t` forces 64-bit offsets even when unnecessary.

## 6. Alternatives if 64-bit sorting is slow

### 6.1 First choice: keep `uint64_t` but set `end_bit = B`

This is the recommended first optimization and is not a different sort. If the Morton key
uses only `B` significant bits, call:

```cpp
cub::DeviceRadixSort::SortPairs(
    d_temp, temp_bytes, d_keys, d_values, M,
    0,        // begin_bit
    B,        // end_bit: only the used Morton bits
    stream);
```

CUB then runs `ceil(B/8)` 8-bit passes instead of 8. For a 3D Morton key with
`b_d = ceil(log2(nbins_d))` bits per dimension, `B = b_x + b_y + b_z` (plus any
dimensionality-specific bits). On the T1000, `B <= 30` gave 4 passes and was ~2x faster
than `end_bit = 64` in the probe. The key type stays `uint64_t`, so the same code handles
larger grids on bigger GPUs up to 64 bits. Note that `end_bit` only reduces the number of
passes; the key array is still 8 bytes per item, and CUB still reads/writes 8-byte keys
on every pass. Ensure bits above `B` are zero (or mask the keys); CUB sorts only the
`[begin_bit, end_bit)` range and ignores higher bits (`device_radix_sort.cuh:246-248`,
`:323-326`, `:396-398`, `:1030-1036`). For `B = 64`, `end_bit = 64` and the mask is a
no-op; avoid an undefined `1ULL << 64` in hand-written masking code.

### 6.2 If `B <= 32`: a `uint32_t` key

Using `uint32_t` keys halves the key bytes moved and the key buffer footprint. On a 4 GB
T1000, any 3D fine grid that fits has at most ~`4 GB / 8 B` ≈ 5e8 complex cells, so the
sum of per-dimension fine-bin bits is at most about 29. The source-bin count also includes
ghost-bin padding, so the actual `B` can be a few bits larger; for the usual
output-tile/bin-size choices it is still <= 32, but ISSUE-008 must compute `B` from the
chosen `numbins` rather than assume it. If `B > 32`, use the 64-bit path. The measured
benefit of a `uint32_t` key over a 64-bit key with the same `end_bit` was small (~7% at
1M, within noise at 4M), so use `uint32_t` only if key memory or the last few percent
matter; do not complicate the code for it initially.

A 32-bit key also limits 3D to roughly 10 bits per dimension (3*10 = 30; 3*11 = 33 > 32).
With `B <= 32` this is always valid; if the grid needs more than 32 total Morton bits, use
the 64-bit path.

### 6.3 Two-pass 32-bit key/payload sort

A stable LSD two-pass scheme is possible:

1. sort `(low32, point_index)` by `low32` with `SortPairs<uint32_t, uint32_t>`;
2. form `high32` for each point (either keep a parallel high-key array and permute it, or
   gather from the original key array using the sorted point index);
3. stable-sort `(high32, point_index)` by `high32`.

For `B > 32`, this is `4 + ceil((B - 32)/8) = ceil(B/8)` radix passes plus a
gather/permute, exactly the same pass count as the one-pass 64-bit sort; it only reduces
bytes per pass. The probe's equal-pass comparison showed key width is a minor effect at
these sizes, so two-pass is unlikely to beat `uint64_t` + `end_bit = B` on the T1000. It
also adds application-level complexity and an extra failure surface. CUB does not provide
a device-wide split-key sort; this would be app code. Defer it unless the prototype
measures the 64-bit sort as the dominant cost and `B > 32` (so the `uint32_t` shortcut is
unavailable).

### 6.4 Rejected alternatives

- **Pack key + payload into one 64-bit key and use `SortKeys<uint64_t>`**: if the Morton
  key needs `B` bits and the point index needs 32 bits, the packed key needs `B+32` bits
  and is sorted as a full 64-bit key (8 passes for `B+32 > 32`), while separate pairs with
  `end_bit=B` need only `ceil(B/8)` passes. For the T1000's `B <= 30`, packing is worse.
- **`cub::DeviceMergeSort::StableSortPairs`**: stable, but CUB documents merge sort as
  slower than radix for arithmetic keys (`device_merge_sort.cuh:31-36`). Not a speed
  alternative.
- **`thrust::sort_by_key` / `thrust::stable_sort_by_key`**: these wrap CUB but add their
  own allocator/dispatch policy; direct CUB gives explicit stream/temp control for this
  prototype.

### 6.5 If `B > 64`

A single `uint64_t` cannot hold more than 64 Morton bits. Options are a hierarchical
two-level key, a 128-bit custom key with a CUB decomposer (CUB's own tests exercise
128-bit custom keys), or a different spatial ordering. This is outside the T1000
prototype's likely bit budget; resolve it only if ISSUE-008 selects a grid that needs it.

## 7. Recommendation for ISSUE-010

Use the **legacy `cub::DeviceRadixSort::SortPairs<uint64_t, uint32_t>` `DoubleBuffer` API**:

- Keep the Morton key as `uint64_t`; no decomposer.
- Compute `B` from the chosen source-bin geometry (ISSUE-008) and pass `end_bit = B`,
  `begin_bit = 0`. Mask the key to `B` bits before sorting.
- Allocate two key arrays and two value arrays in the plan (or as reusable scratch), wrap
  them in `cub::DoubleBuffer`, and consume `d_keys.Current()` /
  `d_values.Current()` after the sort. This avoids the pointer API's extra `N` key/value
  buffers; CUB's remaining temp is about 0.37 B/item at N = 1M, not zero.
  `DoubleBuffer` takes raw pointers, so pass `dethrust(...)`/`.data().get()`; CUB's device
  radix sort does not accept Thrust iterators. If reusing the same `DoubleBuffer` for a
  new batch, write the new input into its `Current()` buffer (or reset `selector`) and
  ensure the alternate buffer is not still being consumed by downstream work.
- Query temp storage once per plan shape with `d_temp_storage = nullptr`; allocate it with
  the plan's `ThrustAllocatorAsync<std::byte>` / `gpu_scratch<std::byte>` on
  `d_plan.stream`. Reuse it across transforms with the same `M`/`B`. Do not query or
  allocate per batch.
- Check the `cudaError_t` return; do not add a stream synchronization just to check the
  sort. Add one only where the algorithm already needs a host-visible result.
- For the first prototype, do not implement the two-pass split or the 32-bit-key path.
  Measure the sort separately with `end_bit = B`; if the sort is still a bottleneck,
  then try `uint32_t` keys (if `B <= 32`) and compare.
- If using the env API instead, guard it with `CUB_VERSION >= 300400` (3.4.0 introduced
  it; the fetched tag is 3.4.2), or tighten the CMake `VERSION` floor, because the
  project currently only requires CCCL major 3 and a system CCCL 3.0/3.1 would not have
  it. Include `<cub/version.cuh>` explicitly for the guard; `<cub/cub.cuh>` does not pull
  in the version macro. The legacy API exists in both the toolkit's CUB 2.7 and fetched
  CUB 3.4.2.
- Add correctness coverage for: empty input, all-equal keys (`B = 0`, skip the sort),
  duplicate keys (stability), clustered points, and a mix of Morton bins. Report `M`,
  `B`, pass count, temp bytes, and sort time separately from the spread.

## 8. Exact probe evidence and source index

### 8.1 Compile/run probes (all under `/tmp`, no repo build changes)

| File | Purpose |
|---|---|
| `/tmp/issue005_probe.cu` | `CUB_VERSION`, API instantiation, temp-storage queries, 16-byte payload, duplicate-key stability |
| `/tmp/issue005_bench.cu` | 64-bit vs 32-bit and `end_bit` timing on the T1000 |
| `/tmp/issue005_env.cu`, `/tmp/issue005_env2.cu` | env API compile/run check (separate buffers in `env2`) |
| `/tmp/cccl_find_test2/CMakeLists.txt` | confirms toolkit CCCL 2.7 is rejected for a major-3 `find_package` request (`CCCL_FOUND=0`) |

Compile check used the exact CCCL 3.4.2 tag headers, CUDA 12.8.61, `-arch=sm_75`, C++17,
and FINUFFT's `--extended-lambda --extra-device-vectorization` flags. It completed with no
errors. A minimal `CUB_VERSION` print produced `300402`.

Temp-storage probe output (n = 1,048,576, key/value = 8/4 bytes, `DoubleBuffer` default
`end_bit = 64`):

```text
pointer SortPairs query:  err=0 bytes=12973567 (12.37 B/item)
DoubleBuffer SortPairs:   err=0 bytes=390655  (0.37 B/item)
StableSortPairs (merge):  err=0 bytes=12585471 (12.00 B/item)
SegmentedRadixSort pairs: err=0 bytes=12583167 (12.00 B/item)
16-byte-payload pairs:    err=0 bytes=26014207
```

The `pointer` query is exactly `12 * n + DoubleBuffer` bytes for this configuration,
confirming that the pointer API's extra `N` key + `N` value buffers are the difference.

### 8.2 CCCL 3.4.2 paths cited (inspected tree `/tmp/cccl-3.4.2`)

- `cub/cub/version.cuh:37` — `CUB_VERSION 300402`.
- `cub/cub/device/device_radix_sort.cuh`
  - `:80-86` supported types; `:113-119` unsigned keys direct;
  - `:131-136` stability;
  - `:230-250` pointer API, `2N` auxiliary storage, `DoubleBuffer` temp note;
  - `:299-302` null-temp size query;
  - `:334-345` pointer `SortPairs` signature;
  - `:374-398` env API version note, no overlap, bit range;
  - `:450-463` env `SortPairs` signature;
  - `:939-941`, `:1086-1088` `DoubleBuffer` buffers may be altered, no in-place;
  - `:1017-1025` selector/current output docs;
  - `:1041-1057` `DoubleBuffer` `SortPairs` signature and `is_overwrite_okay = true`.
- `cub/cub/device/dispatch/dispatch_radix_sort.cuh`
  - `:562-573` onesweep temp-storage formula;
  - `:582-586` null-temp early return;
  - `:1035-1043` empty/no-bit handling;
  - `:1060-1073` single-tile threshold and onesweep dispatch.
- `cub/cub/device/dispatch/tuning/tuning_radix_sort.cuh`
  - `:1332-1356` `Policy700` constants, onesweep 8-bit, items/thread;
  - `:1393-1402` single-tile policy (256×19);
  - `:1889-1909` sm_7x selector, onesweep, `RADIX_RANK_MATCH_EARLY_COUNTS_ANY`.
- `cub/cub/util_type.cuh:745-774` — `cub::DoubleBuffer`, `Current()` / `Alternate()`.
- `cub/cub/detail/choose_offset.cuh:34-43` — 32-bit vs 64-bit `OffsetT`.
- `cub/cub/detail/env_dispatch.cuh:40-75` — env query/allocate/execute/deallocate.
- `cub/cub/detail/device_memory_resource.cuh:31-94` — default `cudaMallocAsync` /
  `cudaFreeAsync` resource.
- `cub/cub/detail/temporary_storage.cuh:350-378` — env allocation calls
  `mr.allocate(stream, ...)` / `mr.deallocate(stream, ...)`.
- `cub/test/catch2_test_device_radix_sort_pairs.cu:28-29` — custom value type test.
- `c2h/include/c2h/custom_type.h:13-17` — custom type has two `size_t` members.
- `libcudacxx/include/cuda/__memory_resource/resource.h:53-78` — `cuda::mr::resource`
  concept.
- `README.md:112-139` — header-only, `-I` not `-isystem`.
- `README.md:241-247` — CCCL 3.x supports CUDA 12.x/13.x.
- `cmake/CCCLCheckCudaArchitectures.cmake:17-18` — minimum arch 75.

### 8.3 FINUFFT paths cited

- `CMakeLists.txt:83-84` — `CUDA11_CCCL_VERSION`, `CUDA12_CCCL_VERSION 3.4.2`.
- `cmake/setupCCCL.cmake:19-36` — major-3 `cpmfindpackage` and fetched tag.
- `src/cuda/CMakeLists.txt:101-129` — C++17 and `CCCL::CCCL` interface link.
- `include/cufinufft/cufinufft_plan_t.hpp:140-203` — `ThrustAllocatorAsync`,
  `gpu_scratch`.
- `include/cufinufft/types.hpp:17` — `CUFINUFFT_BIGINT = int`.
- `include/cufinufft/types.hpp:61-85` — `GpuCapabilities`, `memory_pools_supported`.
- `src/cuda/makeplan.cu:261-265` — pool-support warning.

### 8.4 Installed toolkit paths cited

- `/home/paseo/cuda-12.8/include/cub/version.cuh:61` — `CUB_VERSION 200700`.
- `/home/paseo/cuda-12.8/lib64/cmake/cccl/cccl-config-version.cmake:1-17` — CCCL 2.7.0,
  incompatible with major-3 request.
- `/home/paseo/cuda-12.8/version.json` — `cuda_cccl` `12.8.55`.
- `/home/paseo/cuda-12.8/include/cub/device/device_radix_sort.cuh` — toolkit CUB 2.7 has
  the legacy `SortPairs` API but not the 3.4 env API.

## 9. Open / deferred

- The exact Morton bit budget `B` and source-bin geometry are ISSUE-008 decisions; this
  note only supplies the CUB cost model and the formula `B = sum_d b_d`.
- The microbenchmark is synthetic and random-key; Morton keys are not random (many
  points share bins), so histogram skew may change CUB's sort time. ISSUE-010 must
  measure the real point distribution, `M`, `B`, and end-to-end spread cost.
- Cross-GPU behavior (A100/H100/Ada) is not measured. The 64-bit-with-`end_bit` approach
  is portable; the 32-bit-key shortcut depends on the grid bit budget and device memory.
- If the project ever needs more than 64 Morton bits, the 128-bit custom-key decomposer
  path should be evaluated separately.
