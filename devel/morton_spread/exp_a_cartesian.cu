// Experiment A: output-cell-owned 3D type-1 spreading with Cartesian point bins.
//
// Throwaway prototype for the wayfinder ticket
// docs/issues/in-progress/ISSUE-009-output-cell-ownership-cartesian-prototype.md.
// It is not part of the library build and changes nothing under src/ or include/.
//
// Ownership: each fine-grid cell is written by exactly one thread, once, at the end
// (no global atomics, no shared-memory accumulation, no merge).
// Query structure: points are counting-sorted by Cartesian query bin (qbin, edge qb
// cells); each thread owns a 2x2x2 group of cells inside a 16x16x8 tile and scans the
// query bins that intersect its group expanded by ceil(ns/2)=3 cells.
//
// Kernel values and coordinates are taken from cuFINUFFT itself
// (include/cufinufft/spreadinterp.hpp: fold_rescale + eval_kernel_vec_horner), so the
// prototype differs from method 3 only in the accumulation strategy.
//
// Build: see README.md / run_exp_a.sh.
// Usage: exp_a <M> <N1> <N2> <N3> <cluster> <qb> <nrun>

#include <cuda_runtime.h>
#include <cufinufft.h>
#include <cub/device/device_radix_sort.cuh>
#include <cufinufft/spreadinterp.hpp>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#define CK(x)                                                                          \
  do {                                                                                 \
    int _e = (x);                                                                      \
    if (_e) {                                                                          \
      std::fprintf(stderr, "%s failed: %d\n", #x, _e);                                 \
      std::exit(1);                                                                    \
    }                                                                                  \
  } while (0)
#define CU(x)                                                                          \
  do {                                                                                 \
    cudaError_t _e = (x);                                                              \
    if (_e != cudaSuccess) {                                                           \
      std::fprintf(stderr, "%s failed: %s\n", #x, cudaGetErrorString(_e));             \
      std::exit(1);                                                                    \
    }                                                                                  \
  } while (0)

static constexpr double PI = 3.141592653589793238462643383279502884;

// Fixed prototype geometry (from ISSUE-008, plus the qbin refinement):
//   tile 16x16x8, one CUDA threadblock per tile, 256 threads, 2x2x2 cells per thread.
static constexpr int SUPPORT_WIDTH      = 6;  // kernel width, ns=6 for tol 1e-5 at sigma=2
static constexpr int OUTPUT_TILE_X  = 16;
static constexpr int OUTPUT_TILE_Y  = 16;
static constexpr int OUTPUT_TILE_Z  = 8;
static constexpr int THREADS_PER_BLOCK = 128;                 // 4 warps, 8x8x8 warp region
static constexpr int WARPS_PER_BLOCK  = THREADS_PER_BLOCK / 32;
static constexpr int QUERY_BIN_HALO    = (SUPPORT_WIDTH + 1) / 2;        // 3
static constexpr int GROUP_SIZE_X     = 2;                   // cells per thread per dim
static constexpr int GROUP_SIZE_Y     = 2;
static constexpr int GROUP_SIZE_Z     = 4;                   // 16 cells per thread

__host__ __device__ __forceinline__ int floordiv(int a, int b) {
  // Truncating division + a mask correction: no predicate, no branch.
  const int q = a / b;
  const int r = a - q * b;  // same sign as a
  return q - ((r >> 31) & 1);
}

// ---------------------------------------------------------------------------
// Preprocessing: Cartesian query bins -> (start, count) table + sorted indices.
// ---------------------------------------------------------------------------

__global__ void kernel_compute_qbin(const float *__restrict__ x, const float *__restrict__ y,
                                    const float *__restrict__ z, int M, int nf1, int nf2,
                                    int nf3, int qb, int nqx, int nqy, int nqz,
                                    int *__restrict__ qbin) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
    int b[3];
    const float co[3] = {x[i], y[i], z[i]};
    const int nf[3]   = {nf1, nf2, nf3};
    const int nq[3]   = {nqx, nqy, nqz};
    for (int d = 0; d < 3; d++) {
      float r = cufinufft::spreadinterp::fold_rescale(co[d], nf[d]);
      b[d]    = (int)(r / (float)qb);
      b[d]    = min(max(b[d], 0), nq[d] - 1);
    }
    qbin[i] = b[0] + nqx * (b[1] + nqy * b[2]);
  }
}

__global__ void kernel_hist(const int *__restrict__ qbin, int M, int *__restrict__ counts) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x)
    atomicAdd(&counts[qbin[i]], 1);
}

// Radix-sort preprocessing: fill 0..M-1, gather the sorted payload, mark bin starts.
__global__ void kernel_fill_idx(int *__restrict__ idx, int M) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x)
    idx[i] = i;
}

__global__ void kernel_gather_sorted(const int *__restrict__ perm,
                                     const float *__restrict__ x, const float *__restrict__ y,
                                     const float *__restrict__ z,
                                     const cuFloatComplex *__restrict__ c,
                                     float *__restrict__ sx, float *__restrict__ sy,
                                     float *__restrict__ sz, cuFloatComplex *__restrict__ sc,
                                     int M) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
    const int p = perm[i];
    sx[i]       = x[p];
    sy[i]       = y[p];
    sz[i]       = z[p];
    sc[i]       = c[p];
  }
}

// mark start[b] = first sorted position of bin b; empty bins keep the -1 sentinel.
__global__ void kernel_mark_starts(const unsigned int *__restrict__ keys, int M,
                                   int *__restrict__ start) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x)
    if (i == 0 || keys[i] != keys[i - 1]) start[keys[i]] = i;
}

__global__ void kernel_scatter(const int *__restrict__ qbin, int M,
                               int *__restrict__ cursor, const int *__restrict__ binstart,
                               int *__restrict__ sorted_idx, const float *__restrict__ x,
                               const float *__restrict__ y, const float *__restrict__ z,
                               const cuFloatComplex *__restrict__ c,
                               float *__restrict__ sx, float *__restrict__ sy,
                               float *__restrict__ sz, cuFloatComplex *__restrict__ sc) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
    const int b   = qbin[i];
    const int pos = atomicAdd(&cursor[b], 1);
    const int dst = binstart[b] + pos;
    sorted_idx[dst] = i;
    sx[dst]         = x[i];
    sy[dst]         = y[i];
    sz[dst]         = z[i];
    sc[dst]         = c[i];
  }
}

// ---------------------------------------------------------------------------
// Statistics kernel: candidate point reads (warp-level point reread factor).
// One thread per warp region, mirrors the spread kernel's bin enumeration.
// ---------------------------------------------------------------------------
__global__ void kernel_visits(const int *__restrict__ binstart, int nqx, int nqy, int nqz,
                              int nf1, int nf2, int nf3, int qb, int tiles_x,
                              int tiles_y, unsigned long long *visits,
                              unsigned long long *active_warps) {
  const int nwarps = tiles_x * tiles_y * ((nf3 + OUTPUT_TILE_Z - 1) / OUTPUT_TILE_Z) * WARPS_PER_BLOCK;
  const int wid    = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane   = threadIdx.x & 31;
  if (wid >= nwarps) return;
  const int warp = wid % WARPS_PER_BLOCK;
  const int tile = wid / WARPS_PER_BLOCK;
  const int tz   = tile / (tiles_x * tiles_y);
  const int ty   = (tile / tiles_x) % tiles_y;
  const int tx   = tile % tiles_x;
  const int X0   = tx * OUTPUT_TILE_X + (warp & 1) * 8;
  const int Y0   = ty * OUTPUT_TILE_Y + ((warp >> 1) & 1) * 8;
  const int Z0   = tz * OUTPUT_TILE_Z + (warp >> 2) * 8;
  const int bx0 = floordiv(X0 - QUERY_BIN_HALO, qb), bx1 = (X0 + 7 + QUERY_BIN_HALO) / qb;
  const int by0 = floordiv(Y0 - QUERY_BIN_HALO, qb), by1 = (Y0 + 7 + QUERY_BIN_HALO) / qb;
  const int bz0 = floordiv(Z0 - QUERY_BIN_HALO, qb), bz1 = (Z0 + 7 + QUERY_BIN_HALO) / qb;

  // One warp cooperates on one region. Flatten the bin cube and give each lane
  // a strided subset; the original single-thread version left the GPU almost idle.
  const int nx  = bx1 - bx0 + 1;
  const int ny  = by1 - by0 + 1;
  const int nz  = bz1 - bz0 + 1;
  const int ynz = ny * nz;
  const int total = nx * ynz;
  unsigned long long v = 0;
  for (int idx = lane; idx < total; idx += 32) {
    int t = idx;
    const int ix = t / ynz;
    t -= ix * ynz;
    const int iy = t / nz;
    const int iz = t - iy * nz;
    int wx = (bx0 + ix) % nqx;
    wx += (wx >> 31) & nqx;
    int wy = (by0 + iy) % nqy;
    wy += (wy >> 31) & nqy;
    int wz = (bz0 + iz) % nqz;
    wz += (wz >> 31) & nqz;
    const int bin = wx + nqx * (wy + nqy * wz);
    v += (unsigned long long)(binstart[bin + 1] - binstart[bin]);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
  if (lane == 0) {
    visits[wid]       = v;
    active_warps[wid] = (v != 0) ? 1ULL : 0ULL;
  }
}

// Compact active tiles in tile order. Single-threaded on purpose: the tile count is
// small (128 for 64^3, 1024 for 128^3) and this keeps the list deterministic without
// atomics. The metadata cost is measured separately by the host.
__global__ void kernel_compact_tiles(const unsigned long long *__restrict__ active_warps,
                                     int ntiles, int *__restrict__ tile_ids,
                                     int *__restrict__ tile_count) {
  int n = 0;
  for (int tile = 0; tile < ntiles; ++tile) {
    unsigned long long active = 0;
#pragma unroll
    for (int w = 0; w < WARPS_PER_BLOCK; ++w) active |= active_warps[tile * WARPS_PER_BLOCK + w];
    if (active) tile_ids[n++] = tile;
  }
  tile_count[0] = n;
}

// ---------------------------------------------------------------------------
// Output-cell-owned spread kernel (warp-cooperative gather).
//
// Block = 256 threads = 8 warps, one tile (16x16x8) per block.
// Warp region = 8x8x4 cells; lane -> 2x2x2 group (lane grid 4x4x2).
// The warp enumerates the query bins covering its region expanded by QUERY_BIN_HALO and
// stages each point cooperatively: 18 lanes evaluate one Horner coefficient
// each, one lane records the support start, one lane records the strength.
// Staged points are in shared memory, so every lane reads the same bytes
// (broadcast) and can index the kernel values dynamically at no local-memory
// cost. Each lane accumulates its own 8 cells and writes them exactly once.
// ---------------------------------------------------------------------------
__device__ __constant__ float KERNEL_COEFFICIENT_TABLE[9][6] = {
    {2.0875119883113444E-03, 1.5741818314646622E-01, 8.2446837122968719E-01,
     8.2446837122968786E-01, 1.5741818314646619E-01, 2.0875119883208707E-03},
    {7.2383827471879060E-03, 2.0903648995439425E-01, 3.2052935784357611E-01,
     -3.2052935784357595E-01, -2.0903648995439431E-01, -7.2383827471776199E-03},
    {1.0180085126333447E-02, 9.2337811484268992E-02, -1.0253741712233815E-01,
     -1.0253741712233826E-01, 9.2337811484268992E-02, 1.0180085126343148E-02},
    {7.3669955501269495E-03, 4.9102900025223368E-03, -5.1302324979469585E-02,
     5.1302324979469453E-02, -4.9102900025223412E-03, -7.3669955501178205E-03},
    {2.7444270008043898E-03, -8.0004810696545064E-03, 5.2920367975572286E-03,
     5.2920367975572606E-03, -8.0004810696545012E-03, 2.7444270008144416E-03},
    {3.2622379114949579E-04, -1.8514138516535774E-03, 3.8520985619444484E-03,
     -3.8520985619445560E-03, 1.8514138516535451E-03, -3.2622379114026316E-04},
    {-1.2239646122606804E-04, 2.2750660293439196E-04, -1.2702072030321002E-04,
     -1.2702072030313421E-04, 2.2750660293438364E-04, -1.2239646121695212E-04},
    {-4.6695893922773253E-05, 1.1717219021522327E-04, -1.8098268625867290E-04,
     1.8098268625856663E-04, -1.1717219021521502E-04, 4.6695893931712934E-05},
    {-1.5875418082722892E-06, 7.2147850129502829E-07, -7.0930078290747281E-08,
     -7.0930078325137639E-08, 7.2147850125572373E-07, -1.5875417996328233E-06}};

static constexpr int MAX_POINTS_PER_BATCH = 14;
static constexpr int VALUES_PER_POINT = 28;  // 3 support starts + 2 strengths + 18 coefficients + 5 zero pads
static constexpr int BLOCK_POINTS_PER_BATCH = 56;

// Raw pass: lanes 0..staged_point_count-1 issue all the global loads for the batch at once, so
// the warp waits on one round of coalesced loads instead of ~staged_point_count serialized ones.
// Coordinates and strength land in slots 0..4 of each staged point record.
__device__ __forceinline__ void stage_raw_points(
    int first_point_index, int staged_point_count, int lane_index,
    const float *__restrict__ input_coordinate_x,
    const float *__restrict__ input_coordinate_y,
    const float *__restrict__ input_coordinate_z,
    const float2 *__restrict__ input_strength_pairs,
    float *__restrict__ destination_batch_values) {
  if (lane_index < staged_point_count) {
    const int point_index     = first_point_index + lane_index;
    float *const staged_point = destination_batch_values + lane_index * VALUES_PER_POINT;
    staged_point[0]           = __ldg(input_coordinate_x + point_index);
    staged_point[1]           = __ldg(input_coordinate_y + point_index);
    staged_point[2]           = __ldg(input_coordinate_z + point_index);
    const float2 point_strength = __ldg(input_strength_pairs + point_index);
    staged_point[3]           = point_strength.x;
    staged_point[4]           = point_strength.y;
  }
}

// Coefficient pass for one staged point: the 18 active lanes evaluate one Horner
// coefficient each (nine polynomial coefficients preloaded in registers); the six
// lanes of a dimension fold the same broadcast coordinate in parallel, and the
// support start is written by all six lanes of that dimension (same value, so no
// branch is needed). The lane geometry (fine_grid_size_dimension, support offset, coefficient offset) is
// loop-invariant and supplied by the caller.
__device__ __forceinline__ void prepare_point_coefficients(
    float *__restrict__ staged_point, int fine_grid_size_dimension,
    int support_start_slot, int coefficient_slot,
    const float *__restrict__ kernel_coefficients) {
  using cufinufft::spreadinterp::fold_rescale;
  const float folded_coordinate =
      fold_rescale(staged_point[support_start_slot], fine_grid_size_dimension);
  const float support_start_float =
      ceilf(folded_coordinate - (float)(SUPPORT_WIDTH / 2));
  const int support_start_int = (int)support_start_float;
  const float horner_argument =
      __fmaf_rn(2.f, support_start_float - folded_coordinate,
                (float)(SUPPORT_WIDTH - 1));
  reinterpret_cast<int *>(staged_point)[support_start_slot] = support_start_int;
  float coefficient_value = kernel_coefficients[8];
#pragma unroll
  for (int horner_term = 7; horner_term >= 0; horner_term--)
    coefficient_value = __fmaf_rn(horner_argument, coefficient_value,
                                  kernel_coefficients[horner_term]);
  staged_point[coefficient_slot] = coefficient_value;
}

// Support placement of a point in the frame of a GROUP_SIZE-cell group starting at lane_group_start.
// Branchless: group_cell_count = number of group cells covered (0,1,2); the covered cells are
// contiguous, local cell local_cell_offset+k gets kernel coefficient k[support_coefficient_offset+k] for k < group_cell_count.
__device__ __forceinline__ void place_support(int support_start, int fine_grid_size_dimension, int lane_group_start, int group_size, int &group_cell_count,
                                            int &support_coefficient_offset, int &local_cell_offset) {
  int relative_support_start = support_start - lane_group_start;
  relative_support_start += fine_grid_size_dimension & (relative_support_start >> 31);       // relative_support_start < 0 -> relative_support_start + fine_grid_size_dimension; relative_support_start is now < fine_grid_size_dimension for lane_group_start >= 0,
                              // so the former relative_support_start >= fine_grid_size_dimension correction cannot fire.
  // group_cell_count/support_coefficient_offset/local_cell_offset below are selects: ptxas lowers them to SEL/ISETP with no branch
  // (SASS branch count is identical to an explicit mask form, which measured
  // 11-17% slower because it adds ~21 LOP3 per scan).
  const bool below_group  = relative_support_start < group_size;
  const bool wraps_high = relative_support_start >= fine_grid_size_dimension - (SUPPORT_WIDTH - 1);
  const int wrapped_cell_count    = SUPPORT_WIDTH - (fine_grid_size_dimension - relative_support_start);
  group_cell_count  = below_group ? (group_size - relative_support_start) : (wraps_high ? (wrapped_cell_count < group_size ? wrapped_cell_count : group_size) : 0);
  support_coefficient_offset = below_group ? 0 : (wraps_high ? (fine_grid_size_dimension - relative_support_start) : 0);
  local_cell_offset = below_group ? relative_support_start : 0;
}

// Fast path for a support interval and a lane group that do not wrap around the
// periodic fine-grid boundary. The overlap is a plain interval intersection, so
// no modulo and no below_group/wraps_high predicates are needed.
__device__ __forceinline__ void place_support_nowrap(int support_start, int lane_group_start, int group_size, int &group_cell_count,
                                                   int &support_coefficient_offset, int &local_cell_offset) {
  const int intersection_begin = max(support_start, lane_group_start);
  const int intersection_end =
      min(support_start + SUPPORT_WIDTH, lane_group_start + group_size);
  group_cell_count = max(intersection_end - intersection_begin, 0);
  support_coefficient_offset = intersection_begin - support_start;
  local_cell_offset = intersection_begin - lane_group_start;
}

// One staged point: support placement, kernel values, 16-cell accumulation.
// Pulled out of the loop so the scan can issue two points back to back (ILP).
template <bool SUPPORTS_DO_NOT_WRAP>
__device__ __forceinline__ void scan_one_staged_point(const float *__restrict__ staged_point, int fine_grid_size_x, int fine_grid_size_y,
                                           int fine_grid_size_z, int lane_group_x_start, int lane_group_y_start, int lane_group_z_start,
                                           float2 (&output_cell_accumulators)[16]) {
  const int2 support_start_xy = *reinterpret_cast<const int2 *>(staged_point);  // support starts x,y as ints
  const int support_start_z   = reinterpret_cast<const int *>(staged_point)[2];
  const int support_start_x = support_start_xy.x, support_start_y = support_start_xy.y;
  const float strength_real = staged_point[3], strength_imag = staged_point[4];
  int group_cell_count_x, support_coefficient_offset_x, local_cell_offset_x, group_cell_count_y, support_coefficient_offset_y, local_cell_offset_y, group_cell_count_z, support_coefficient_offset_z, local_cell_offset_z;
  if constexpr (SUPPORTS_DO_NOT_WRAP) {
    place_support_nowrap(support_start_x, lane_group_x_start, GROUP_SIZE_X, group_cell_count_x, support_coefficient_offset_x, local_cell_offset_x);
    place_support_nowrap(support_start_y, lane_group_y_start, GROUP_SIZE_Y, group_cell_count_y, support_coefficient_offset_y, local_cell_offset_y);
    place_support_nowrap(support_start_z, lane_group_z_start, GROUP_SIZE_Z, group_cell_count_z, support_coefficient_offset_z, local_cell_offset_z);
  } else {
    place_support(support_start_x, fine_grid_size_x, lane_group_x_start, GROUP_SIZE_X, group_cell_count_x, support_coefficient_offset_x, local_cell_offset_x);
    place_support(support_start_y, fine_grid_size_y, lane_group_y_start, GROUP_SIZE_Y, group_cell_count_y, support_coefficient_offset_y, local_cell_offset_y);
    place_support(support_start_z, fine_grid_size_z, lane_group_z_start, GROUP_SIZE_Z, group_cell_count_z, support_coefficient_offset_z, local_cell_offset_z);
  }
  if (!(group_cell_count_x && group_cell_count_y && group_cell_count_z)) return;
  // Padded coefficient rows: the slot just past a real coefficient is zero, so
  // these loads never need the group_cell_count_x/group_cell_count_y/group_cell_count_z range selects.
  const float x_kernel_value_0 = staged_point[5 + support_coefficient_offset_x], x_kernel_value_1 = staged_point[6 + support_coefficient_offset_x];
  const float y_kernel_value_0 = staged_point[12 + support_coefficient_offset_y], y_kernel_value_1 = staged_point[13 + support_coefficient_offset_y];
  const float z_kernel_value_0 = staged_point[19 + support_coefficient_offset_z], z_kernel_value_1 = staged_point[20 + support_coefficient_offset_z];
  const float z_kernel_value_2 = staged_point[21 + support_coefficient_offset_z], z_kernel_value_3 = staged_point[22 + support_coefficient_offset_z];
  const float x_kernel_values[2] = {(local_cell_offset_x == 0) ? x_kernel_value_0 : 0.f, (local_cell_offset_x == 0) ? x_kernel_value_1 : x_kernel_value_0};
  const float y_kernel_values[2] = {(local_cell_offset_y == 0) ? y_kernel_value_0 : 0.f, (local_cell_offset_y == 0) ? y_kernel_value_1 : y_kernel_value_0};
  const float z_kernel_values[4] = {(local_cell_offset_z == 0) ? z_kernel_value_0 : 0.f,
                       (local_cell_offset_z == 0) ? z_kernel_value_1 : ((local_cell_offset_z == 1) ? z_kernel_value_0 : 0.f),
                       (local_cell_offset_z == 0) ? z_kernel_value_2 : ((local_cell_offset_z == 1) ? z_kernel_value_1 : ((local_cell_offset_z == 2) ? z_kernel_value_0 : 0.f)),
                       (local_cell_offset_z == 0) ? z_kernel_value_3 : ((local_cell_offset_z == 1) ? z_kernel_value_2
                                          : ((local_cell_offset_z == 2) ? z_kernel_value_1 : ((local_cell_offset_z == 3) ? z_kernel_value_0 : 0.f)))};
  // Indexed by local_x + 2*local_y to match the accumulator layout.
  const float xy_weights[4] = {x_kernel_values[0] * y_kernel_values[0], x_kernel_values[1] * y_kernel_values[0], x_kernel_values[0] * y_kernel_values[1], x_kernel_values[1] * y_kernel_values[1]};
#pragma unroll
  for (int local_z = 0; local_z < 4; local_z++) {
#pragma unroll
    for (int local_y = 0; local_y < 2; local_y++) {
#pragma unroll
      for (int local_x = 0; local_x < 2; local_x++) {
        const int output_cell_index    = local_x + 2 * local_y + 4 * local_z;
        const float weight = xy_weights[local_x + 2 * local_y] * z_kernel_values[local_z];
        output_cell_accumulators[output_cell_index].x += strength_real * weight;
        output_cell_accumulators[output_cell_index].y += strength_imag * weight;
      }
    }
  }
}

__global__ void spread_cartesian_owned(
    const float *__restrict__ input_coordinate_x,
    const float *__restrict__ input_coordinate_y,
    const float *__restrict__ input_coordinate_z,
    const cuFloatComplex *__restrict__ input_strengths,
    const int *__restrict__ query_bin_point_offsets,
    cuFloatComplex *__restrict__ output_grid, int fine_grid_size_x, int fine_grid_size_y,
    int fine_grid_size_z, int query_bin_count_x, int query_bin_count_y, int query_bin_count_z, int query_bin_size, int output_tile_count_x, int output_tile_count_y, int split_count,
    const int *__restrict__ active_tile_indices) {
  __shared__ __align__(16) float
      staged_point_values[WARPS_PER_BLOCK][2][MAX_POINTS_PER_BATCH][VALUES_PER_POINT];

  const int warp_index     = threadIdx.x >> 5;
  const int lane_index     = threadIdx.x & 31;
  const int active_tile_list_index = blockIdx.x / split_count;
  const int output_tile_index     = active_tile_indices[active_tile_list_index];
  const int split_index    = blockIdx.x - active_tile_list_index * split_count;
  const int tile_z_index   = output_tile_index / (output_tile_count_x * output_tile_count_y);
  const int tile_y_index   = (output_tile_index / output_tile_count_x) % output_tile_count_y;
  const int tile_x_index   = output_tile_index % output_tile_count_x;
  const int warp_region_x_start   = tile_x_index * OUTPUT_TILE_X + (warp_index & 1) * 8;
  const int warp_region_y_start   = tile_y_index * OUTPUT_TILE_Y + ((warp_index >> 1) & 1) * 8;
  const int warp_region_z_start   = tile_z_index * OUTPUT_TILE_Z + (warp_index >> 2) * 8;
  const bool warp_region_inside_grid = (warp_region_x_start + 7 < fine_grid_size_x && warp_region_y_start + 7 < fine_grid_size_y && warp_region_z_start + 7 < fine_grid_size_z);
  const int lane_group_x_start  = warp_region_x_start + (lane_index & 3) * GROUP_SIZE_X;
  const int lane_group_y_start  = warp_region_y_start + ((lane_index >> 2) & 3) * GROUP_SIZE_Y;
  const int lane_group_z_start  = warp_region_z_start + ((lane_index >> 4) & 1) * GROUP_SIZE_Z;
  // No early return: __syncwarp() must be executed by the whole warp. Groups
  // outside the grid (partial edge tiles) accumulate garbage but never store.

  float2 output_cell_accumulators[16];
#pragma unroll
  for (int output_cell_index = 0; output_cell_index < 16; output_cell_index++)
    output_cell_accumulators[output_cell_index] = make_float2(0.f, 0.f);

  // The nine Horner coefficients and the staging geometry for this lane are
  // loop-invariant: load/compute them once instead of per point.
  float kernel_coefficients[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const int dimension   = lane_index / 6;
  const int kernel_coefficient_index   = lane_index % 6;
  const int fine_grid_size_dimension  = (dimension == 0) ? fine_grid_size_x : ((dimension == 1) ? fine_grid_size_y : fine_grid_size_z);
  const int support_start_slot = dimension;
  const int coefficient_slot = 5 + dimension * 7 + kernel_coefficient_index;
  if (lane_index < 18) {
#pragma unroll
    for (int kernel_term = 0; kernel_term < 9; kernel_term++)
      kernel_coefficients[kernel_term] =
          KERNEL_COEFFICIENT_TABLE[kernel_term][kernel_coefficient_index];
  }

  // Zero the padded coefficient slots once in both warp-private buffers. The
  // padding is never written again; scan_one_staged_point relies on it staying zero.
  for (int buffered_point_slot = lane_index;
       buffered_point_slot < 2 * MAX_POINTS_PER_BATCH;
       buffered_point_slot += 32) {
    float *const padded_staged_point =
        &staged_point_values[warp_index][0][0][0] +
        buffered_point_slot * VALUES_PER_POINT;
    padded_staged_point[11] = 0.f;
    padded_staged_point[18] = 0.f;
    padded_staged_point[25] = 0.f;
    padded_staged_point[26] = 0.f;
    padded_staged_point[27] = 0.f;
  }
  __syncwarp();

  const int query_bin_min_x = floordiv(warp_region_x_start - QUERY_BIN_HALO, query_bin_size), query_bin_max_x = (warp_region_x_start + 7 + QUERY_BIN_HALO) / query_bin_size;
  const int query_bin_min_y = floordiv(warp_region_y_start - QUERY_BIN_HALO, query_bin_size), query_bin_max_y = (warp_region_y_start + 7 + QUERY_BIN_HALO) / query_bin_size;
  const int query_bin_min_z = floordiv(warp_region_z_start - QUERY_BIN_HALO, query_bin_size), query_bin_max_z = (warp_region_z_start + 7 + QUERY_BIN_HALO) / query_bin_size;
  const float2 *const input_strength_pairs =
      reinterpret_cast<const float2 *>(input_strengths);

  // Walk the warp's bin cube as contiguous x-runs. Bins along x are consecutive in the
  // sorted point table, so a whole x-run is one contiguous point range and can be batched
  // in 16s regardless of the query-bin size (this is what makes query_bin_size=1, reread 5.36, usable).
  const int warp_query_bin_count_x = min(query_bin_max_x - query_bin_min_x + 1, query_bin_count_x);
  int wrapped_first_query_bin_x         = query_bin_min_x % query_bin_count_x;
  wrapped_first_query_bin_x += (wrapped_first_query_bin_x >> 31) & query_bin_count_x;
  const int wrapped_last_query_bin_x = wrapped_first_query_bin_x + warp_query_bin_count_x - 1;
  for (int query_bin_y = query_bin_min_y; query_bin_y <= query_bin_max_y; query_bin_y++) {
    int wrapped_query_bin_y = query_bin_y % query_bin_count_y;
    wrapped_query_bin_y += (wrapped_query_bin_y >> 31) & query_bin_count_y;
    for (int query_bin_z = query_bin_min_z; query_bin_z <= query_bin_max_z; query_bin_z++) {
      int wrapped_query_bin_z = query_bin_z % query_bin_count_z;
      wrapped_query_bin_z += (wrapped_query_bin_z >> 31) & query_bin_count_z;
      const int query_bin_base_index = query_bin_count_x * (wrapped_query_bin_y + query_bin_count_y * wrapped_query_bin_z);
      for (int x_segment_index = 0; x_segment_index < 2; ++x_segment_index) {
        // The x-run can wrap around the periodic boundary: two contiguous segments.
        const int segment_first_bin_x = (x_segment_index == 0) ? wrapped_first_query_bin_x : 0;
        const int segment_last_bin_x = (x_segment_index == 0) ? min(wrapped_last_query_bin_x, query_bin_count_x - 1) : (wrapped_last_query_bin_x - query_bin_count_x);
        if (segment_last_bin_x < segment_first_bin_x) continue;
        const int point_range_begin  = query_bin_point_offsets[query_bin_base_index + segment_first_bin_x];
        const int point_range_end  = query_bin_point_offsets[query_bin_base_index + segment_last_bin_x + 1];
        const int point_batch_stride = split_count * MAX_POINTS_PER_BATCH;
        int first_point_index         = point_range_begin + split_index * MAX_POINTS_PER_BATCH;
        if (first_point_index < point_range_end) {
          int staged_point_count    = min(MAX_POINTS_PER_BATCH, point_range_end - first_point_index);
          float *current_batch_values = &staged_point_values[warp_index][0][0][0];
          float *next_batch_values = &staged_point_values[warp_index][1][0][0];
          stage_raw_points(first_point_index, staged_point_count, lane_index,
                            input_coordinate_x, input_coordinate_y, input_coordinate_z,
                            input_strength_pairs, current_batch_values);
          __syncwarp();
          for (;;) {
            // Hoist the lane_index predicate out of the per-point coefficient loop: the
            // 18 staging lanes run the whole batch, then all lanes meet at the sync.
            if (lane_index < 18) {
              float *staged_point = current_batch_values;
              for (int staged_point_slot = 0; staged_point_slot < staged_point_count; staged_point_slot++, staged_point += VALUES_PER_POINT)
                prepare_point_coefficients(staged_point, fine_grid_size_dimension, support_start_slot, coefficient_slot, kernel_coefficients);
            }
            __syncwarp();
            const int next_first_point_index   = first_point_index + point_batch_stride;
            const int next_staged_point_count = (next_first_point_index < point_range_end) ? min(MAX_POINTS_PER_BATCH, point_range_end - next_first_point_index) : 0;
            if (next_staged_point_count)
              stage_raw_points(next_first_point_index, next_staged_point_count, lane_index,
                              input_coordinate_x, input_coordinate_y, input_coordinate_z,
                              input_strength_pairs, next_batch_values);

            // One warp-wide decision per batch: if every staged point has a
            // non-wrapping support and the warp region is interior, the whole
            // batch uses the branch-free min/max support path. Otherwise the
            // batch uses the general periodic path.
            bool support_inside_grid = true;
            if (lane_index < staged_point_count) {
              const int *const support_start_bits = reinterpret_cast<const int *>(
                  current_batch_values + lane_index * VALUES_PER_POINT);
              const int support_start_x = support_start_bits[0];
              const int support_start_y = support_start_bits[1];
              const int support_start_z = support_start_bits[2];
              support_inside_grid = support_start_x >= 0 && support_start_y >= 0 && support_start_z >= 0 &&
                             support_start_x + SUPPORT_WIDTH <= fine_grid_size_x && support_start_y + SUPPORT_WIDTH <= fine_grid_size_y && support_start_z + SUPPORT_WIDTH <= fine_grid_size_z;
            }
            const bool whole_batch_nonwrapping = warp_region_inside_grid && __all_sync(0xffffffff, support_inside_grid);

            int staged_point_slot = 0;
            if (whole_batch_nonwrapping) {
              for (; staged_point_slot + 1 < staged_point_count; staged_point_slot += 2) {
                scan_one_staged_point<true>(current_batch_values + staged_point_slot * VALUES_PER_POINT, fine_grid_size_x, fine_grid_size_y, fine_grid_size_z, lane_group_x_start, lane_group_y_start, lane_group_z_start, output_cell_accumulators);
                scan_one_staged_point<true>(current_batch_values + (staged_point_slot + 1) * VALUES_PER_POINT, fine_grid_size_x, fine_grid_size_y, fine_grid_size_z, lane_group_x_start, lane_group_y_start, lane_group_z_start, output_cell_accumulators);
              }
              if (staged_point_slot < staged_point_count)
                scan_one_staged_point<true>(current_batch_values + staged_point_slot * VALUES_PER_POINT, fine_grid_size_x, fine_grid_size_y, fine_grid_size_z, lane_group_x_start, lane_group_y_start, lane_group_z_start, output_cell_accumulators);
            } else {
              for (; staged_point_slot + 1 < staged_point_count; staged_point_slot += 2) {
                scan_one_staged_point<false>(current_batch_values + staged_point_slot * VALUES_PER_POINT, fine_grid_size_x, fine_grid_size_y, fine_grid_size_z, lane_group_x_start, lane_group_y_start, lane_group_z_start, output_cell_accumulators);
                scan_one_staged_point<false>(current_batch_values + (staged_point_slot + 1) * VALUES_PER_POINT, fine_grid_size_x, fine_grid_size_y, fine_grid_size_z, lane_group_x_start, lane_group_y_start, lane_group_z_start, output_cell_accumulators);
              }
              if (staged_point_slot < staged_point_count)
                scan_one_staged_point<false>(current_batch_values + staged_point_slot * VALUES_PER_POINT, fine_grid_size_x, fine_grid_size_y, fine_grid_size_z, lane_group_x_start, lane_group_y_start, lane_group_z_start, output_cell_accumulators);
            }
            __syncwarp();
            if (!next_staged_point_count) break;
            first_point_index  = next_first_point_index;
            staged_point_count = next_staged_point_count;
            float *temporary_batch_values = current_batch_values;
            current_batch_values        = next_batch_values;
            next_batch_values        = temporary_batch_values;
          }
        }
      }
    }
  }

  // Unique, non-atomic store into this split's slice (split_index 0 only when split_count==1).
  const long long total_fine_grid_cells = (long long)fine_grid_size_x * fine_grid_size_y * fine_grid_size_z;
  cuFloatComplex *const split_output_grid = output_grid + (long long)split_index * total_fine_grid_cells;
#pragma unroll
  for (int output_cell_index = 0; output_cell_index < 16; output_cell_index++) {
    const int local_cell_offset_x = output_cell_index & 1;
    const int local_cell_offset_y = (output_cell_index >> 1) & 1;
    const int local_cell_offset_z = (output_cell_index >> 2) & 3;
    const long long global_cell_x = lane_group_x_start + local_cell_offset_x;
    const long long global_cell_y = lane_group_y_start + local_cell_offset_y;
    const long long global_cell_z = lane_group_z_start + local_cell_offset_z;
    if (global_cell_x < fine_grid_size_x &&
        global_cell_y < fine_grid_size_y &&
        global_cell_z < fine_grid_size_z) {
      const long long flat_output_index =
          global_cell_x + (long long)fine_grid_size_x *
                              (global_cell_y +
                               (long long)fine_grid_size_y * global_cell_z);
      split_output_grid[flat_output_index] =
          make_cuFloatComplex(output_cell_accumulators[output_cell_index].x,
                              output_cell_accumulators[output_cell_index].y);
    }
  }
}


// Block-wide shared staging for the split (clustered) path. Each of the four warps
// stages a contiguous quarter of the block batch; the block then scans the whole batch
// once per warp. The next batch is staged before scanning the current one, so the
// coefficient work is amortized across all four warp regions without a separate phase.
__device__ __forceinline__ void stage_block_quarter(
    int first_point_index, int staged_point_count, int warp_index, int lane_index,
    const float *__restrict__ input_coordinate_x,
    const float *__restrict__ input_coordinate_y,
    const float *__restrict__ input_coordinate_z,
    const float2 *__restrict__ input_strength_pairs,
    float *__restrict__ destination_batch_values, int fine_grid_size_dimension,
    int support_start_slot, int coefficient_slot,
    const float *__restrict__ kernel_coefficients) {
  const int begin = (staged_point_count * warp_index) / WARPS_PER_BLOCK;
  const int end = (staged_point_count * (warp_index + 1)) / WARPS_PER_BLOCK;
  const int count = end - begin;
  if (count <= 0) return;
  stage_raw_points(first_point_index + begin, count, lane_index, input_coordinate_x,
                   input_coordinate_y, input_coordinate_z, input_strength_pairs,
                   destination_batch_values + begin * VALUES_PER_POINT);
  __syncwarp();
  if (lane_index < 18) {
    float *staged_point = destination_batch_values + begin * VALUES_PER_POINT;
    for (int i = 0; i < count; i++, staged_point += VALUES_PER_POINT)
      prepare_point_coefficients(staged_point, fine_grid_size_dimension, support_start_slot,
                                 coefficient_slot, kernel_coefficients);
  }
  __syncwarp();
}

// Block-shared variant: one 16x16x8 tile per block, four warp regions, one shared
// double-buffered batch. All warps scan the block's union point stream, so the scan
// work matches the per-warp variant when the block reread is 4x the warp reread
// (the clustered case), while stage_coeff/stage_raw are computed once per block point.
__global__ void spread_cartesian_owned_block_shared(
    const float *__restrict__ input_coordinate_x,
    const float *__restrict__ input_coordinate_y,
    const float *__restrict__ input_coordinate_z,
    const cuFloatComplex *__restrict__ input_strengths,
    const int *__restrict__ query_bin_point_offsets,
    cuFloatComplex *__restrict__ output_grid, int fine_grid_size_x, int fine_grid_size_y,
    int fine_grid_size_z, int query_bin_count_x, int query_bin_count_y,
    int query_bin_count_z, int query_bin_size, int output_tile_count_x,
    int output_tile_count_y, int split_count,
    const int *__restrict__ active_tile_indices) {
  __shared__ __align__(16) float
      staged_point_values[2][BLOCK_POINTS_PER_BATCH][VALUES_PER_POINT];

  const int warp_index = threadIdx.x >> 5;
  const int lane_index = threadIdx.x & 31;
  const int active_tile_list_index = blockIdx.x / split_count;
  const int output_tile_index = active_tile_indices[active_tile_list_index];
  const int split_index = blockIdx.x - active_tile_list_index * split_count;
  const int tile_z_index = output_tile_index / (output_tile_count_x * output_tile_count_y);
  const int tile_y_index = (output_tile_index / output_tile_count_x) % output_tile_count_y;
  const int tile_x_index = output_tile_index % output_tile_count_x;
  const int block_x_start = tile_x_index * OUTPUT_TILE_X;
  const int block_y_start = tile_y_index * OUTPUT_TILE_Y;
  const int block_z_start = tile_z_index * OUTPUT_TILE_Z;
  const int warp_region_x_start = block_x_start + (warp_index & 1) * 8;
  const int warp_region_y_start = block_y_start + ((warp_index >> 1) & 1) * 8;
  const int warp_region_z_start = block_z_start;
  const bool warp_region_inside_grid =
      (warp_region_x_start + 7 < fine_grid_size_x &&
       warp_region_y_start + 7 < fine_grid_size_y &&
       warp_region_z_start + 7 < fine_grid_size_z);
  const int lane_group_x_start = warp_region_x_start + (lane_index & 3) * GROUP_SIZE_X;
  const int lane_group_y_start = warp_region_y_start + ((lane_index >> 2) & 3) * GROUP_SIZE_Y;
  const int lane_group_z_start = warp_region_z_start + ((lane_index >> 4) & 1) * GROUP_SIZE_Z;

  float2 output_cell_accumulators[16];
#pragma unroll
  for (int output_cell_index = 0; output_cell_index < 16; output_cell_index++)
    output_cell_accumulators[output_cell_index] = make_float2(0.f, 0.f);

  float kernel_coefficients[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  const int dimension = lane_index / 6;
  const int kernel_coefficient_index = lane_index % 6;
  const int fine_grid_size_dimension =
      (dimension == 0) ? fine_grid_size_x : ((dimension == 1) ? fine_grid_size_y : fine_grid_size_z);
  const int support_start_slot = dimension;
  const int coefficient_slot = 5 + dimension * 7 + kernel_coefficient_index;
  if (lane_index < 18) {
#pragma unroll
    for (int kernel_term = 0; kernel_term < 9; kernel_term++)
      kernel_coefficients[kernel_term] =
          KERNEL_COEFFICIENT_TABLE[kernel_term][kernel_coefficient_index];
  }

  // Zero the pads in both shared buffers once. The block, not an individual warp,
  // owns these buffers, so all threads cooperatively initialize them.
  for (int buffered_point_slot = threadIdx.x;
       buffered_point_slot < 2 * BLOCK_POINTS_PER_BATCH;
       buffered_point_slot += blockDim.x) {
    const int buffer = buffered_point_slot / BLOCK_POINTS_PER_BATCH;
    const int point_slot = buffered_point_slot % BLOCK_POINTS_PER_BATCH;
    float *const padded_staged_point = &staged_point_values[buffer][point_slot][0];
    padded_staged_point[11] = 0.f;
    padded_staged_point[18] = 0.f;
    padded_staged_point[25] = 0.f;
    padded_staged_point[26] = 0.f;
    padded_staged_point[27] = 0.f;
  }
  __syncthreads();

  const int query_bin_min_x =
      floordiv(block_x_start - QUERY_BIN_HALO, query_bin_size);
  const int query_bin_max_x =
      (block_x_start + OUTPUT_TILE_X - 1 + QUERY_BIN_HALO) / query_bin_size;
  const int query_bin_min_y =
      floordiv(block_y_start - QUERY_BIN_HALO, query_bin_size);
  const int query_bin_max_y =
      (block_y_start + OUTPUT_TILE_Y - 1 + QUERY_BIN_HALO) / query_bin_size;
  const int query_bin_min_z =
      floordiv(block_z_start - QUERY_BIN_HALO, query_bin_size);
  const int query_bin_max_z =
      (block_z_start + OUTPUT_TILE_Z - 1 + QUERY_BIN_HALO) / query_bin_size;
  const float2 *const input_strength_pairs =
      reinterpret_cast<const float2 *>(input_strengths);

  const int warp_query_bin_count_x =
      min(query_bin_max_x - query_bin_min_x + 1, query_bin_count_x);
  int wrapped_first_query_bin_x = query_bin_min_x % query_bin_count_x;
  wrapped_first_query_bin_x += (wrapped_first_query_bin_x >> 31) & query_bin_count_x;
  const int wrapped_last_query_bin_x =
      wrapped_first_query_bin_x + warp_query_bin_count_x - 1;
  for (int query_bin_y = query_bin_min_y; query_bin_y <= query_bin_max_y; query_bin_y++) {
    int wrapped_query_bin_y = query_bin_y % query_bin_count_y;
    wrapped_query_bin_y += (wrapped_query_bin_y >> 31) & query_bin_count_y;
    for (int query_bin_z = query_bin_min_z; query_bin_z <= query_bin_max_z; query_bin_z++) {
      int wrapped_query_bin_z = query_bin_z % query_bin_count_z;
      wrapped_query_bin_z += (wrapped_query_bin_z >> 31) & query_bin_count_z;
      const int query_bin_base_index =
          query_bin_count_x * (wrapped_query_bin_y + query_bin_count_y * wrapped_query_bin_z);
      for (int x_segment_index = 0; x_segment_index < 2; ++x_segment_index) {
        const int segment_first_bin_x =
            (x_segment_index == 0) ? wrapped_first_query_bin_x : 0;
        const int segment_last_bin_x =
            (x_segment_index == 0)
                ? min(wrapped_last_query_bin_x, query_bin_count_x - 1)
                : (wrapped_last_query_bin_x - query_bin_count_x);
        if (segment_last_bin_x < segment_first_bin_x) continue;
        const int point_range_begin =
            query_bin_point_offsets[query_bin_base_index + segment_first_bin_x];
        const int point_range_end =
            query_bin_point_offsets[query_bin_base_index + segment_last_bin_x + 1];
        const int point_batch_stride = split_count * BLOCK_POINTS_PER_BATCH;
        int first_point_index = point_range_begin + split_index * BLOCK_POINTS_PER_BATCH;
        if (first_point_index >= point_range_end) continue;

        int staged_point_count =
            min(BLOCK_POINTS_PER_BATCH, point_range_end - first_point_index);
        float *current_batch_values = &staged_point_values[0][0][0];
        float *next_batch_values = &staged_point_values[1][0][0];
        stage_block_quarter(first_point_index, staged_point_count, warp_index, lane_index,
                            input_coordinate_x, input_coordinate_y, input_coordinate_z,
                            input_strength_pairs, current_batch_values,
                            fine_grid_size_dimension, support_start_slot, coefficient_slot,
                            kernel_coefficients);
        __syncthreads();

        for (;;) {
          const int next_first_point_index = first_point_index + point_batch_stride;
          const int next_staged_point_count =
              (next_first_point_index < point_range_end)
                  ? min(BLOCK_POINTS_PER_BATCH, point_range_end - next_first_point_index)
                  : 0;
          if (next_staged_point_count)
            stage_block_quarter(next_first_point_index, next_staged_point_count, warp_index,
                                lane_index, input_coordinate_x, input_coordinate_y,
                                input_coordinate_z, input_strength_pairs,
                                next_batch_values, fine_grid_size_dimension,
                                support_start_slot, coefficient_slot,
                                kernel_coefficients);

          bool support_inside_grid = true;
          if (lane_index < staged_point_count) {
            const int *const support_start_bits =
                reinterpret_cast<const int *>(current_batch_values) +
                lane_index * VALUES_PER_POINT;
            const int support_start_x = support_start_bits[0];
            const int support_start_y = support_start_bits[1];
            const int support_start_z = support_start_bits[2];
            support_inside_grid =
                support_start_x >= 0 && support_start_y >= 0 && support_start_z >= 0 &&
                support_start_x + SUPPORT_WIDTH <= fine_grid_size_x &&
                support_start_y + SUPPORT_WIDTH <= fine_grid_size_y &&
                support_start_z + SUPPORT_WIDTH <= fine_grid_size_z;
          }
          const bool whole_batch_nonwrapping =
              warp_region_inside_grid && __all_sync(0xffffffff, support_inside_grid);

          int staged_point_slot = 0;
          if (whole_batch_nonwrapping) {
            for (; staged_point_slot + 1 < staged_point_count; staged_point_slot += 2) {
              scan_one_staged_point<true>(
                  current_batch_values + staged_point_slot * VALUES_PER_POINT,
                  fine_grid_size_x, fine_grid_size_y, fine_grid_size_z,
                  lane_group_x_start, lane_group_y_start, lane_group_z_start,
                  output_cell_accumulators);
              scan_one_staged_point<true>(
                  current_batch_values + (staged_point_slot + 1) * VALUES_PER_POINT,
                  fine_grid_size_x, fine_grid_size_y, fine_grid_size_z,
                  lane_group_x_start, lane_group_y_start, lane_group_z_start,
                  output_cell_accumulators);
            }
            if (staged_point_slot < staged_point_count)
              scan_one_staged_point<true>(
                  current_batch_values + staged_point_slot * VALUES_PER_POINT,
                  fine_grid_size_x, fine_grid_size_y, fine_grid_size_z,
                  lane_group_x_start, lane_group_y_start, lane_group_z_start,
                  output_cell_accumulators);
          } else {
            for (; staged_point_slot + 1 < staged_point_count; staged_point_slot += 2) {
              scan_one_staged_point<false>(
                  current_batch_values + staged_point_slot * VALUES_PER_POINT,
                  fine_grid_size_x, fine_grid_size_y, fine_grid_size_z,
                  lane_group_x_start, lane_group_y_start, lane_group_z_start,
                  output_cell_accumulators);
              scan_one_staged_point<false>(
                  current_batch_values + (staged_point_slot + 1) * VALUES_PER_POINT,
                  fine_grid_size_x, fine_grid_size_y, fine_grid_size_z,
                  lane_group_x_start, lane_group_y_start, lane_group_z_start,
                  output_cell_accumulators);
            }
            if (staged_point_slot < staged_point_count)
              scan_one_staged_point<false>(
                  current_batch_values + staged_point_slot * VALUES_PER_POINT,
                  fine_grid_size_x, fine_grid_size_y, fine_grid_size_z,
                  lane_group_x_start, lane_group_y_start, lane_group_z_start,
                  output_cell_accumulators);
          }

          __syncthreads();
          if (!next_staged_point_count) break;
          first_point_index = next_first_point_index;
          staged_point_count = next_staged_point_count;
          float *temporary_batch_values = current_batch_values;
          current_batch_values = next_batch_values;
          next_batch_values = temporary_batch_values;
        }
      }
    }
  }

  const long long total_fine_grid_cells =
      (long long)fine_grid_size_x * fine_grid_size_y * fine_grid_size_z;
  cuFloatComplex *const split_output_grid =
      output_grid + (long long)split_index * total_fine_grid_cells;
#pragma unroll
  for (int output_cell_index = 0; output_cell_index < 16; output_cell_index++) {
    const int local_cell_offset_x = output_cell_index & 1;
    const int local_cell_offset_y = (output_cell_index >> 1) & 1;
    const int local_cell_offset_z = (output_cell_index >> 2) & 3;
    const long long global_cell_x = lane_group_x_start + local_cell_offset_x;
    const long long global_cell_y = lane_group_y_start + local_cell_offset_y;
    const long long global_cell_z = lane_group_z_start + local_cell_offset_z;
    if (global_cell_x < fine_grid_size_x && global_cell_y < fine_grid_size_y &&
        global_cell_z < fine_grid_size_z) {
      const long long flat_output_index =
          global_cell_x + (long long)fine_grid_size_x *
                              (global_cell_y + (long long)fine_grid_size_y * global_cell_z);
      split_output_grid[flat_output_index] =
          make_cuFloatComplex(output_cell_accumulators[output_cell_index].x,
                              output_cell_accumulators[output_cell_index].y);
    }
  }
}


// Sum only the cells of active tiles. Inactive tiles are zeroed once by the host.
// This cuts the split-reduction traffic for clustered inputs from nsplit*full-grid
// to nsplit*active-tile-cells (16/128 tiles in the mandatory clustered case).
__global__ void kernel_sum_splits_active(
    const cuFloatComplex *__restrict__ part, int nsplit, long long nf,
    cuFloatComplex *__restrict__ fw, const int *__restrict__ tile_ids, int tile_count,
    int tiles_x, int tiles_y, int nf1, int nf2, int nf3) {
  const int cells_per_tile = OUTPUT_TILE_X * OUTPUT_TILE_Y * OUTPUT_TILE_Z;
  const long long total_cells = (long long)tile_count * cells_per_tile;
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total_cells;
       i += (long long)blockDim.x * gridDim.x) {
    const int tile_slot = (int)(i / cells_per_tile);
    const int local_cell = (int)(i - (long long)tile_slot * cells_per_tile);
    const int tile = tile_ids[tile_slot];
    const int tz = tile / (tiles_x * tiles_y);
    const int ty = (tile / tiles_x) % tiles_y;
    const int tx = tile % tiles_x;
    const int local_x = local_cell % OUTPUT_TILE_X;
    const int local_y = (local_cell / OUTPUT_TILE_X) % OUTPUT_TILE_Y;
    const int local_z = local_cell / (OUTPUT_TILE_X * OUTPUT_TILE_Y);
    const int gx = tx * OUTPUT_TILE_X + local_x;
    const int gy = ty * OUTPUT_TILE_Y + local_y;
    const int gz = tz * OUTPUT_TILE_Z + local_z;
    if (gx >= nf1 || gy >= nf2 || gz >= nf3) continue;
    const long long idx = (long long)gx + (long long)nf1 * (gy + (long long)nf2 * gz);
    const float2 *const p = reinterpret_cast<const float2 *>(part);
    float2 acc = p[idx];
    for (int k = 1; k < nsplit; ++k) {
      const float2 v = p[(long long)k * nf + idx];
      acc.x += v.x;
      acc.y += v.y;
    }
    fw[idx] = make_cuFloatComplex(acc.x, acc.y);
  }
}

// Sum the per-split partial grids into the final grid (only used when nsplit > 1).
__global__ void kernel_sum_splits(const cuFloatComplex *__restrict__ part, int nsplit,
                                  long long nf, cuFloatComplex *__restrict__ fw) {
  for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < nf;
       i += (long long)blockDim.x * gridDim.x) {
    const float2 *const p = reinterpret_cast<const float2 *>(part);
    float2 acc            = p[i];
    for (int k = 1; k < nsplit; ++k) {
      const float2 v = p[(long long)k * nf + i];
      acc.x += v.x;
      acc.y += v.y;
    }
    fw[i] = make_cuFloatComplex(acc.x, acc.y);
  }
}


// Block-level candidate-read count for the active tile list. One warp per active tile;
// the bin enumeration mirrors the block-shared spread kernel's x-run walk, so the sum
// is exactly the number of point stages a block-shared launch would perform.
__global__ void kernel_block_visits(
    const int *__restrict__ binstart, int nqx, int nqy, int nqz, int qb,
    int tiles_x, int tiles_y, const int *__restrict__ tile_ids, int tile_count,
    unsigned long long *__restrict__ block_visits) {
  const int wid = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (wid >= tile_count) return;
  const int tile = tile_ids[wid];
  const int tz = tile / (tiles_x * tiles_y);
  const int ty = (tile / tiles_x) % tiles_y;
  const int tx = tile % tiles_x;
  const int block_x_start = tx * OUTPUT_TILE_X;
  const int block_y_start = ty * OUTPUT_TILE_Y;
  const int block_z_start = tz * OUTPUT_TILE_Z;
  const int bx0 = floordiv(block_x_start - QUERY_BIN_HALO, qb);
  const int bx1 = (block_x_start + OUTPUT_TILE_X - 1 + QUERY_BIN_HALO) / qb;
  const int by0 = floordiv(block_y_start - QUERY_BIN_HALO, qb);
  const int by1 = (block_y_start + OUTPUT_TILE_Y - 1 + QUERY_BIN_HALO) / qb;
  const int bz0 = floordiv(block_z_start - QUERY_BIN_HALO, qb);
  const int bz1 = (block_z_start + OUTPUT_TILE_Z - 1 + QUERY_BIN_HALO) / qb;
  const int nx = min(bx1 - bx0 + 1, nqx);
  int wlo = bx0 % nqx;
  wlo += (wlo >> 31) & nqx;
  const int wend = wlo + nx - 1;
  const int ny = by1 - by0 + 1;
  const int nz = bz1 - bz0 + 1;
  const int ynz = ny * nz;
  unsigned long long v = 0;
  for (int idx = lane; idx < ynz; idx += 32) {
    int t = idx;
    const int iy = t / nz;
    const int iz = t - iy * nz;
    int wy = (by0 + iy) % nqy;
    wy += (wy >> 31) & nqy;
    int wz = (bz0 + iz) % nqz;
    wz += (wz >> 31) & nqz;
    const int base = nqx * (wy + nqy * wz);
    const int x0 = wlo;
    const int x1 = min(wend, nqx - 1);
    if (x1 >= x0) v += (unsigned long long)(binstart[base + x1 + 1] - binstart[base + x0]);
    const int x2 = 0;
    const int x3 = wend - nqx;
    if (x3 >= x2) v += (unsigned long long)(binstart[base + x3 + 1] - binstart[base + x2]);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
  if (lane == 0) block_visits[wid] = v;
}


// Packed original point for the radix-path gather. The gather is random
// (perm[i]), so reading one 32-byte record avoids four independent random array
// accesses (x/y/z/strength) per point.
struct alignas(32) PackedPoint {
  float x, y, z, strength_real;
  float strength_imag;
  float pad[3];
};

__global__ void kernel_pack_points(const float *__restrict__ x, const float *__restrict__ y,
                                   const float *__restrict__ z,
                                   const cuFloatComplex *__restrict__ c,
                                   float4 *__restrict__ packed, int M) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
    packed[2 * i] = make_float4(x[i], y[i], z[i], c[i].x);
    packed[2 * i + 1] = make_float4(c[i].y, 0.f, 0.f, 0.f);
  }
}

__global__ void kernel_gather_sorted_packed(const int *__restrict__ perm,
                                            const float4 *__restrict__ packed,
                                            float *__restrict__ sx, float *__restrict__ sy,
                                            float *__restrict__ sz,
                                            cuFloatComplex *__restrict__ sc, int M) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
    const int p = perm[i];
    const float4 lo = packed[2 * p];
    const float4 hi = packed[2 * p + 1];
    sx[i] = lo.x;
    sy[i] = lo.y;
    sz[i] = lo.z;
    sc[i] = make_cuFloatComplex(lo.w, hi.x);
  }
}

// Fused qbin + packed-point pass for the radix path. The qbin kernel and the pack
// kernel both read x/y/z; combining them avoids a second read of the coordinate arrays.
__global__ void kernel_compute_qbin_pack(
    const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
    const cuFloatComplex *__restrict__ c, int M, int nf1, int nf2, int nf3, int qb, int nqx,
    int nqy, int nqz, int *__restrict__ qbin, float4 *__restrict__ packed) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < M; i += blockDim.x * gridDim.x) {
    int b[3];
    const float co[3] = {x[i], y[i], z[i]};
    const int nf[3]   = {nf1, nf2, nf3};
    const int nq[3]   = {nqx, nqy, nqz};
    for (int d = 0; d < 3; d++) {
      float r = cufinufft::spreadinterp::fold_rescale(co[d], nf[d]);
      b[d]    = (int)(r / (float)qb);
      b[d]    = min(max(b[d], 0), nq[d] - 1);
    }
    qbin[i] = b[0] + nqx * (b[1] + nqy * b[2]);
    packed[2 * i]     = make_float4(x[i], y[i], z[i], c[i].x);
    packed[2 * i + 1] = make_float4(c[i].y, 0.f, 0.f, 0.f);
  }
}


// Radix-path binstart construction. mark_starts writes the first sorted position of
// each non-empty bin; a reverse exclusive scan fills the empty bins with the next
// non-empty start. The scan operator keeps the carried start unless the current bin
// has one, i.e. next[b] = starts[b] if non-empty else next[b+1].
struct NextStartOp {
  __host__ __device__ int operator()(int carried, int current) const {
    return current >= 0 ? current : carried;
  }
};

__global__ void kernel_reverse_int(const int *__restrict__ src, int *__restrict__ dst,
                                   int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    dst[i] = src[n - 1 - i];
}

__global__ void kernel_fill_next_start(const int *__restrict__ reversed_starts,
                                       const int *__restrict__ reversed_prefix,
                                       int *__restrict__ binstart, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    binstart[n - 1 - i] = reversed_starts[i] >= 0 ? reversed_starts[i] : reversed_prefix[i];
}

// ---------------------------------------------------------------------------
// Host harness
// ---------------------------------------------------------------------------

struct timing {
  double setpts_ms, exec_ms, total_ms;
};

static timing run_cufinufft(int method, int M, int N1, int N2, int N3, float *dx,
                            float *dy, float *dz, cuFloatComplex *dc,
                            cuFloatComplex *dfk, int nrun, bool keep_output) {
  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.gpu_method           = method;
  opts.gpu_sort             = 1;
  opts.gpu_kerevalmeth      = 1;
  opts.gpu_spreadinterponly = 1;
  opts.modeord              = 0;

  int64_t nmodes[3] = {N1, N2, N3};
  cufinufftf_plan plan;
  std::vector<double> sp, ex, tot;

  for (int r = -2; r < nrun; r++) {  // r<0: warm-up
    cudaEvent_t e0, e1, e2;
    CU(cudaEventCreate(&e0));
    CU(cudaEventCreate(&e1));
    CU(cudaEventCreate(&e2));
    char range[64];
    CK(cufinufftf_makeplan(1, 3, nmodes, 1, 1, 1e-5, &plan, &opts));
    CU(cudaEventRecord(e0));
    std::snprintf(range, sizeof(range), "m%d_setpts", method);
    nvtxRangePushA(range);
    CK(cufinufftf_setpts(plan, M, dx, dy, dz, 0, nullptr, nullptr, nullptr));
    nvtxRangePop();
    CU(cudaEventRecord(e1));
    std::snprintf(range, sizeof(range), "m%d_execute", method);
    nvtxRangePushA(range);
    CK(cufinufftf_execute(plan, dc, dfk));
    nvtxRangePop();
    CU(cudaEventRecord(e2));
    CU(cudaEventSynchronize(e2));
    if (r >= 0) {
      float a = 0, b = 0, c = 0;
      CU(cudaEventElapsedTime(&a, e0, e1));
      CU(cudaEventElapsedTime(&b, e1, e2));
      CU(cudaEventElapsedTime(&c, e0, e2));
      sp.push_back(a);
      ex.push_back(b);
      tot.push_back(c);
    }
    CK(cufinufftf_destroy(plan));
    CU(cudaEventDestroy(e0));
    CU(cudaEventDestroy(e1));
    CU(cudaEventDestroy(e2));
  }
  (void)keep_output;
  auto med = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  return {med(sp), med(ex), med(tot)};
}

int main(int argc, char **argv) {
  const int M       = argc > 1 ? std::atoi(argv[1]) : 1000000;
  const int N1      = argc > 2 ? std::atoi(argv[2]) : 64;
  const int N2      = argc > 3 ? std::atoi(argv[3]) : 64;
  const int N3      = argc > 4 ? std::atoi(argv[4]) : 64;
  const int cluster = argc > 5 ? std::atoi(argv[5]) : 0;
  const int qb      = argc > 6 ? std::atoi(argv[6]) : 2;
  const int nrun    = argc > 7 ? std::atoi(argv[7]) : 5;
  const int prep    = argc > 8 ? std::atoi(argv[8]) : 1;  // 0 = counting sort, 1 = radix sort

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  std::vector<float> x(M), y(M), z(M);
  for (int i = 0; i < M; ++i) {
    if (cluster) {
      const int cc = i % 8;
      const float cx = (cc & 1) ? 0.7f : -0.7f;
      const float cy = (cc & 2) ? 0.7f : -0.7f;
      const float cz = (cc & 4) ? 0.7f : -0.7f;
      std::normal_distribution<float> g(0.0f, 0.01f);
      x[i] = PI * (cx + g(rng));
      y[i] = PI * (cy + g(rng));
      z[i] = PI * (cz + g(rng));
    } else {
      x[i] = PI * uni(rng);
      y[i] = PI * uni(rng);
      z[i] = PI * uni(rng);
    }
  }

  float *dx, *dy, *dz;
  cuFloatComplex *dc, *dfk_ref, *dfk_new, *dfk_ctx;
  const long long nf = (long long)N1 * N2 * N3;
  CU(cudaMalloc(&dx, M * sizeof(float)));
  CU(cudaMalloc(&dy, M * sizeof(float)));
  CU(cudaMalloc(&dz, M * sizeof(float)));
  CU(cudaMalloc(&dc, M * sizeof(cuFloatComplex)));
  CU(cudaMalloc(&dfk_ref, nf * sizeof(cuFloatComplex)));
  CU(cudaMalloc(&dfk_new, nf * sizeof(cuFloatComplex)));
  CU(cudaMalloc(&dfk_ctx, nf * sizeof(cuFloatComplex)));
  CU(cudaMemcpy(dx, x.data(), M * sizeof(float), cudaMemcpyHostToDevice));
  CU(cudaMemcpy(dy, y.data(), M * sizeof(float), cudaMemcpyHostToDevice));
  CU(cudaMemcpy(dz, z.data(), M * sizeof(float), cudaMemcpyHostToDevice));
  std::vector<cuFloatComplex> c(M, make_cuFloatComplex(0.1f, 0.2f));
  CU(cudaMemcpy(dc, c.data(), M * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

  // Reference: pinned cuFINUFFT method 3, spread-only, same kernel evaluation.
  const timing t3 = run_cufinufft(3, M, N1, N2, N3, dx, dy, dz, dc, dfk_ref, nrun, true);
  const timing t4 = run_cufinufft(4, M, N1, N2, N3, dx, dy, dz, dc, dfk_ctx, nrun, false);
  const timing t1 = run_cufinufft(1, M, N1, N2, N3, dx, dy, dz, dc, dfk_ctx, nrun, false);

  // ---- preprocessing: Cartesian query bins (timed as a whole) ----
  const int nqx = (N1 + qb - 1) / qb, nqy = (N2 + qb - 1) / qb, nqz = (N3 + qb - 1) / qb;
  const int nbins = nqx * nqy * nqz;
  int end_bit = 1;
  while ((1 << end_bit) < nbins) end_bit++;
  int *d_qbin, *d_counts, *d_cursor, *d_binstart, *d_sorted;
  float *d_sx, *d_sy, *d_sz;
  cuFloatComplex *d_sc;
  unsigned int *d_keys_out = nullptr;
  int *d_perm = nullptr, *d_starts = nullptr, *d_rev_starts = nullptr,
      *d_rev_prefix = nullptr;
  float4 *d_packed = nullptr;
  void *d_sort_tmp = nullptr;
  size_t sort_tmp_bytes = 0;
  void *d_scan_tmp = nullptr;
  size_t scan_tmp_bytes = 0;
  void *d_startscan_tmp = nullptr;
  size_t startscan_tmp_bytes = 0;

  CU(cudaMalloc(&d_qbin, M * sizeof(int)));
  CU(cudaMalloc(&d_counts, (nbins + 1) * sizeof(int)));
  CU(cudaMalloc(&d_cursor, nbins * sizeof(int)));
  CU(cudaMalloc(&d_binstart, (nbins + 1) * sizeof(int)));
  CU(cudaMalloc(&d_sorted, M * sizeof(int)));
  CU(cudaMalloc(&d_sx, M * sizeof(float)));
  CU(cudaMalloc(&d_sy, M * sizeof(float)));
  CU(cudaMalloc(&d_sz, M * sizeof(float)));
  CU(cudaMalloc(&d_sc, M * sizeof(cuFloatComplex)));
  CU(cudaMalloc(&d_keys_out, M * sizeof(unsigned int)));
  CU(cudaMalloc(&d_perm, M * sizeof(int)));
  if (prep == 1) {
    CU(cudaMalloc(&d_packed, (size_t)M * 2 * sizeof(float4)));
    CU(cudaMalloc(&d_starts, nbins * sizeof(int)));
    CU(cudaMalloc(&d_rev_starts, nbins * sizeof(int)));
    CU(cudaMalloc(&d_rev_prefix, nbins * sizeof(int)));
  }
  if (prep == 1) {
    size_t bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, bytes, (const unsigned int *)d_qbin, d_keys_out,
                                    (const unsigned int *)d_sorted, (unsigned int *)d_perm, M,
                                    0, end_bit);
    sort_tmp_bytes = bytes;
    CU(cudaMalloc(&d_sort_tmp, sort_tmp_bytes));
  }
  cub::DeviceScan::ExclusiveSum(nullptr, scan_tmp_bytes, d_counts, d_binstart, nbins + 1);
  CU(cudaMalloc(&d_scan_tmp, scan_tmp_bytes));
  if (prep == 1) {
    NextStartOp next_start_op;
    cub::DeviceScan::ExclusiveScan(nullptr, startscan_tmp_bytes, d_rev_starts, d_rev_prefix,
                                   next_start_op, M, nbins);
    CU(cudaMalloc(&d_startscan_tmp, startscan_tmp_bytes));
  }
  std::vector<cuFloatComplex> h_ref(nf), h_new(nf);

  std::vector<double> pre_ms;
  for (int r = -2; r < nrun; r++) {
    CU(cudaDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    if (prep == 1) {
      nvtxRangePushA("preprocess_qbin_pack");
      kernel_compute_qbin_pack<<<(M + 255) / 256, 256>>>(
          dx, dy, dz, dc, M, N1, N2, N3, qb, nqx, nqy, nqz, d_qbin, d_packed);
      nvtxRangePop();
    } else {
      nvtxRangePushA("preprocess_qbin");
      kernel_compute_qbin<<<(M + 255) / 256, 256>>>(dx, dy, dz, M, N1, N2, N3, qb, nqx, nqy,
                                                    nqz, d_qbin);
      nvtxRangePop();
    }
    if (prep == 1) {
      // CUB radix sort of (bin, point index): no histogram, no per-bin atomics.
      nvtxRangePushA("preprocess_sort");
      kernel_fill_idx<<<(M + 255) / 256, 256>>>(d_sorted, M);
      cub::DeviceRadixSort::SortPairs(d_sort_tmp, sort_tmp_bytes, (const unsigned int *)d_qbin,
                                      d_keys_out, (const unsigned int *)d_sorted,
                                      (unsigned int *)d_perm, M, 0, end_bit);
      kernel_gather_sorted_packed<<<(M + 255) / 256, 256>>>(
          d_perm, d_packed, d_sx, d_sy, d_sz, d_sc, M);
      nvtxRangePop();
      nvtxRangePushA("preprocess_starts");
      CU(cudaMemset(d_starts, 0xFF, nbins * sizeof(int)));
      kernel_mark_starts<<<(M + 255) / 256, 256>>>(d_keys_out, M, d_starts);
      kernel_reverse_int<<<(nbins + 255) / 256, 256>>>(d_starts, d_rev_starts, nbins);
      NextStartOp next_start_op;
      CU(cub::DeviceScan::ExclusiveScan(d_startscan_tmp, startscan_tmp_bytes, d_rev_starts,
                                        d_rev_prefix, next_start_op, M, nbins));
      kernel_fill_next_start<<<(nbins + 255) / 256, 256>>>(d_rev_starts, d_rev_prefix,
                                                            d_binstart, nbins);
      CU(cudaMemcpy(d_binstart + nbins, &M, sizeof(int), cudaMemcpyHostToDevice));
      nvtxRangePop();
    } else {
      // Counting sort: atomic histogram, GPU prefix scan, atomic per-bin scatter.
      CU(cudaMemset(d_counts, 0, (nbins + 1) * sizeof(int)));
      nvtxRangePushA("preprocess_hist");
      kernel_hist<<<(M + 255) / 256, 256>>>(d_qbin, M, d_counts);
      nvtxRangePop();
      nvtxRangePushA("preprocess_scan");
      CU(cub::DeviceScan::ExclusiveSum(d_scan_tmp, scan_tmp_bytes, d_counts, d_binstart,
                                       nbins + 1));
      CU(cudaMemset(d_cursor, 0, nbins * sizeof(int)));
      nvtxRangePop();
      nvtxRangePushA("preprocess_scatter");
      kernel_scatter<<<(M + 255) / 256, 256>>>(d_qbin, M, d_cursor, d_binstart, d_sorted, dx,
                                               dy, dz, dc, d_sx, d_sy, d_sz, d_sc);
      nvtxRangePop();
    }
    CU(cudaGetLastError());
    CU(cudaDeviceSynchronize());
    const auto t1p = std::chrono::steady_clock::now();
    if (r >= 0)
      pre_ms.push_back(std::chrono::duration<double, std::milli>(t1p - t0).count());
  }
  std::sort(pre_ms.begin(), pre_ms.end());
  const double preprocess_ms = pre_ms[pre_ms.size() / 2];

  // ---- new spread kernel ----
  const int tiles_x = (N1 + OUTPUT_TILE_X - 1) / OUTPUT_TILE_X;
  const int tiles_y = (N2 + OUTPUT_TILE_Y - 1) / OUTPUT_TILE_Y;
  const int tiles_z = (N3 + OUTPUT_TILE_Z - 1) / OUTPUT_TILE_Z;
  const int ntiles  = tiles_x * tiles_y * tiles_z;

  // ---- metadata: visits, active warps, active tile list, split decision ----
  // Timed separately as meta_ms; not part of the spread kernel's CUDA-event interval.
  const auto meta_t0 = std::chrono::steady_clock::now();
  const int nwarps_total = ntiles * WARPS_PER_BLOCK;
  unsigned long long *d_visits, *d_active, *d_block_visits;
  int *d_tile_ids = nullptr, *d_tile_count = nullptr;
  CU(cudaMalloc(&d_visits, nwarps_total * sizeof(unsigned long long)));
  CU(cudaMalloc(&d_active, nwarps_total * sizeof(unsigned long long)));
  CU(cudaMalloc(&d_block_visits, ntiles * sizeof(unsigned long long)));
  CU(cudaMalloc(&d_tile_ids, ntiles * sizeof(int)));
  CU(cudaMalloc(&d_tile_count, sizeof(int)));
  cudaEvent_t mv0, mv1, mv2, bv0, bv1;
  CU(cudaEventCreate(&mv0));
  CU(cudaEventCreate(&mv1));
  CU(cudaEventCreate(&mv2));
  CU(cudaEventCreate(&bv0));
  CU(cudaEventCreate(&bv1));
  CU(cudaEventRecord(mv0));
  nvtxRangePushA("kernel_visits");
  kernel_visits<<<(nwarps_total + 7) / 8, 256>>>(d_binstart, nqx, nqy, nqz, N1, N2, N3,
                                                 qb, tiles_x, tiles_y, d_visits, d_active);
  nvtxRangePop();
  CU(cudaEventRecord(mv1));
  nvtxRangePushA("kernel_compact_tiles");
  kernel_compact_tiles<<<1, 1>>>(d_active, ntiles, d_tile_ids, d_tile_count);
  nvtxRangePop();
  CU(cudaEventRecord(mv2));
  int h_tile_count = 0;
  CU(cudaMemcpy(&h_tile_count, d_tile_count, sizeof(int), cudaMemcpyDeviceToHost));
  unsigned long long h_block_visits = 0;
  float blockvisits_ms = 0.f;
  if (h_tile_count > 0) {
    CU(cudaEventRecord(bv0));
    nvtxRangePushA("kernel_block_visits");
    kernel_block_visits<<<(h_tile_count + 7) / 8, 256>>>(
        d_binstart, nqx, nqy, nqz, qb, tiles_x, tiles_y, d_tile_ids, h_tile_count,
        d_block_visits);
    nvtxRangePop();
    CU(cudaEventRecord(bv1));
    std::vector<unsigned long long> hb_block_visits(h_tile_count);
    CU(cudaMemcpy(hb_block_visits.data(), d_block_visits,
                  h_tile_count * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    for (int tile_slot = 0; tile_slot < h_tile_count; tile_slot++)
      h_block_visits += hb_block_visits[tile_slot];
    CU(cudaEventElapsedTime(&blockvisits_ms, bv0, bv1));
  }
  std::vector<unsigned long long> hb_visits(nwarps_total), hb_active(nwarps_total);
  CU(cudaMemcpy(hb_visits.data(), d_visits, nwarps_total * sizeof(unsigned long long),
                cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(hb_active.data(), d_active, nwarps_total * sizeof(unsigned long long),
                cudaMemcpyDeviceToHost));
  unsigned long long h_visits = 0, h_active = 0;
  for (int b = 0; b < nwarps_total; b++) {
    h_visits += hb_visits[b];
    h_active += hb_active[b];
  }
  float visits_ms = 0.f, compact_ms = 0.f;
  CU(cudaEventElapsedTime(&visits_ms, mv0, mv1));
  CU(cudaEventElapsedTime(&compact_ms, mv1, mv2));
  CU(cudaEventDestroy(mv0));
  CU(cudaEventDestroy(mv1));
  CU(cudaEventDestroy(mv2));
  CU(cudaEventDestroy(bv0));
  CU(cudaEventDestroy(bv1));
  // Split only when few warp regions are active (clustered input): raise the number of
  // work items until ~256 warps are in flight. Each split owns a subset of the bins and
  // writes its own partial grid; the reduction sums them.
  // The original target stops clustered at 8 splits (64 active tile-warps * 8 = 512).
  // For low-occupancy clustered inputs, use a higher target so the active tiles are
  // split further; leave uniform/sparse at their existing single-split configuration.
  const long long split_target = (h_active < 512) ? 2048 : 512;
  int nsplit = 1;
  while (nsplit < 32 && (long long)h_active * nsplit < split_target) nsplit <<= 1;
  const char *block_share_env = std::getenv("EXP_A_BLOCK_SHARED");
  // Block sharing is selected from measured candidate reads, not from nsplit or the
  // distribution name: four warps scan each block point, so it is safe only when
  // 4 * block_visits <= warp_visits (5% margin for noise). This enables clustered
  // (8.0M vs 8.0M) and disables uniform (13.2M vs 5.4M) / sparse (18.0M vs 8.0M).
  const bool auto_block_shared =
      h_tile_count > 0 &&
      (unsigned long long)WARPS_PER_BLOCK * h_block_visits * 100 <=
          h_visits * 105;
  const bool use_block_shared =
      block_share_env ? (std::atoi(block_share_env) != 0) : auto_block_shared;
  // nsplit remains only a performance knob. Block-shared staging halves the per-split
  // stage cost, so its measured knee is 56; uniform/sparse remain at nsplit=1.
  if (use_block_shared && nsplit > 1) nsplit = 56;
  const double meta_ms = std::chrono::duration<double, std::milli>(
                             std::chrono::steady_clock::now() - meta_t0)
                             .count();
  cuFloatComplex *dfk_part = nullptr;
  if (nsplit > 1) CU(cudaMalloc(&dfk_part, (long long)nsplit * nf * sizeof(cuFloatComplex)));

  std::vector<double> new_ms;
  cudaEvent_t e0, e1;
  CU(cudaEventCreate(&e0));
  CU(cudaEventCreate(&e1));
  for (int r = -2; r < nrun; r++) {
    CU(cudaEventRecord(e0));
    nvtxRangePushA(nsplit > 1 ? "spread_cartesian_owned_split" : "spread_cartesian_owned");
    if (use_block_shared) {
      spread_cartesian_owned_block_shared<<<h_tile_count * nsplit, THREADS_PER_BLOCK>>>(
          d_sx, d_sy, d_sz, d_sc, d_binstart, nsplit > 1 ? dfk_part : dfk_new, N1, N2, N3,
          nqx, nqy, nqz, qb, tiles_x, tiles_y, nsplit, d_tile_ids);
    } else {
      spread_cartesian_owned<<<h_tile_count * nsplit, THREADS_PER_BLOCK>>>(
          d_sx, d_sy, d_sz, d_sc, d_binstart, nsplit > 1 ? dfk_part : dfk_new, N1, N2, N3, nqx,
          nqy, nqz, qb, tiles_x, tiles_y, nsplit, d_tile_ids);
    }
    if (nsplit > 1) {
      const char *active_reduce_env = std::getenv("EXP_A_ACTIVE_REDUCE");
      const bool active_reduce = active_reduce_env ? (std::atoi(active_reduce_env) != 0) : true;
      if (active_reduce) {
        CU(cudaMemsetAsync(dfk_new, 0, (size_t)nf * sizeof(cuFloatComplex)));
        const long long active_cells = (long long)h_tile_count * OUTPUT_TILE_X *
                                       OUTPUT_TILE_Y * OUTPUT_TILE_Z;
        if (active_cells > 0)
          kernel_sum_splits_active<<<(int)((active_cells + 255) / 256), 256>>>(
              dfk_part, nsplit, nf, dfk_new, d_tile_ids, h_tile_count, tiles_x, tiles_y,
              N1, N2, N3);
      } else {
        kernel_sum_splits<<<(int)((nf + 255) / 256), 256>>>(dfk_part, nsplit, nf, dfk_new);
      }
    }
    nvtxRangePop();
    CU(cudaEventRecord(e1));
    CU(cudaEventSynchronize(e1));
    if (r >= 0) {
      float ms = 0;
      CU(cudaEventElapsedTime(&ms, e0, e1));
      new_ms.push_back(ms);
    }
  }
  std::sort(new_ms.begin(), new_ms.end());
  const double new_ms_med = new_ms[new_ms.size() / 2];

  // ---- correctness ----
  CU(cudaMemcpy(h_ref.data(), dfk_ref, nf * sizeof(cuFloatComplex),
                cudaMemcpyDeviceToHost));
  CU(cudaMemcpy(h_new.data(), dfk_new, nf * sizeof(cuFloatComplex),
                cudaMemcpyDeviceToHost));
  double num = 0, den = 0, maxabs = 0;
  for (long long i = 0; i < nf; i++) {
    const double dr = (double)h_new[i].x - h_ref[i].x;
    const double di = (double)h_new[i].y - h_ref[i].y;
    num += dr * dr + di * di;
    den += (double)h_ref[i].x * h_ref[i].x + (double)h_ref[i].y * h_ref[i].y;
    maxabs = std::max(maxabs, std::max(std::fabs(dr), std::fabs(di)));
  }
  const double relL2 = std::sqrt(num / den);
  // ---- kernel attributes ----
  cudaFuncAttributes attr;
  int blocks_per_sm = 0;
  if (use_block_shared) {
    CU(cudaFuncGetAttributes(&attr, spread_cartesian_owned_block_shared));
    CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, spread_cartesian_owned_block_shared, THREADS_PER_BLOCK, 0));
  } else {
    CU(cudaFuncGetAttributes(&attr, spread_cartesian_owned));
    CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, spread_cartesian_owned,
                                                     THREADS_PER_BLOCK, 0));
  }

  std::printf("case=M%d nf=%dx%dx%d cluster=%d qb=%d\n", M, N1, N2, N3, cluster, qb);
  std::printf("ref   method=3 setpts=%.3f ms exec=%.3f ms total=%.3f ms\n", t3.setpts_ms,
              t3.exec_ms, t3.total_ms);
  std::printf("ctx   method=4 setpts=%.3f ms exec=%.3f ms total=%.3f ms\n", t4.setpts_ms,
              t4.exec_ms, t4.total_ms);
  std::printf("ctx   method=1 setpts=%.3f ms exec=%.3f ms total=%.3f ms\n", t1.setpts_ms,
              t1.exec_ms, t1.total_ms);
  std::printf("new   qbins=%d^3 bins=%d prep=%s preprocess=%.3f ms visits=%.3f ms compact=%.3f ms blockvisits=%.3f ms meta=%.3f ms setup=%.3f ms spread=%.3f ms total=%.3f ms nsplit=%d tiles=%d/%d block_share=%d\n",
              nqx, nbins, prep == 0 ? "count" : "radix", preprocess_ms, visits_ms,
              compact_ms, blockvisits_ms, meta_ms, preprocess_ms + meta_ms, new_ms_med,
              preprocess_ms + meta_ms + new_ms_med, nsplit, h_tile_count, ntiles,
              (int)use_block_shared);
  std::printf("new   speedup_vs_m3_spread=%.2fx spread_ratio=%.3f total_ratio=%.3f\n",
              t3.exec_ms / new_ms_med, new_ms_med / t3.exec_ms,
              (preprocess_ms + meta_ms + new_ms_med) / t3.total_ms);
  std::printf("new   relL2=%.3e maxabs=%.3e visits=%llu reread_factor=%.2f block_visits=%llu block_reread=%.2f active_warps=%llu/%d regs=%d smem=%zu occ_blocks_per_sm=%d\n",
              relL2, maxabs, h_visits, (double)h_visits / (double)M, h_block_visits,
              (double)h_block_visits / (double)M, h_active, nwarps_total,
              attr.numRegs, attr.sharedSizeBytes, blocks_per_sm);
  return 0;
}
