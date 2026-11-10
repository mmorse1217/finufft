// Readable, atomic-free 3D type-1 spreading prototype: implementation.
//
// This is a standalone experiment, not part of the library build. It uses a
// width-6 kernel (sigma=2, tolerance 1e-5), single precision, and periodic
// boundaries. One warp owns one 8x8x8 warp region, and each lane owns a 2x2x4
// lane cell group, so each output cell is written exactly once by one lane.
// There are no spreading atomics.
//
// The launch is split into three phases: plan the active warp regions on the
// host from the bin-offset table (a region is active when any candidate bin is
// non-empty), launch one work item per (active region, split) pair, and (when
// splits are used) reduce the compact partial buffers into the output. A spread
// kernel launch therefore carries a work-item count instead of a threadblock
// tile grid: a full-grid launch passes no active list and indexes flat warp-region
// IDs directly, while an active-list launch decodes
// active_regions[work_id / split_count] and takes
// split_index = work_id % split_count. The split path stores
// partial[(split_index * active_count + region_slot) * WARP_REGION_CELLS +
// local_cell], so it never computes a global cell index. Inactive regions are
// zeroed by the host memset and are never touched by the kernel or the reduction.
//
// Sources are staged in shared memory as one record per source: three support
// starts, six kernel values per dimension, and the strength. One lane stages one
// source, so a 32-source staging group uses every lane of the warp. The hot scan
// places each staged support against the lane group once, then reads the 2/2/4
// covered kernel values as a window. A warp-wide vote selects a non-wrapping
// placement fast path when the warp region is interior and every staged support
// is inside the grid. Grids narrow enough that a support can cover a group prefix
// and suffix at the same time keep the per-cell periodic tap fallback.
// Staging is warp-private: the kernel has no block-wide barrier, only the
// __syncwarp() around the single staging buffer.
//
// For simplicity, binning and sorting are done on the CPU after a GPU kernel
// computes the bin IDs. The latter ensures bin assignment uses the exact same
// GPU fold_rescale as spreading, including at floating-point boundaries.
// This is a correctness/readability prototype, not an optimized implementation.
//
// This translation unit contains only the experiment's kernels and launch
// wrappers. The entry point, timing, reference comparison, and correctness
// checks are in atomic_free_spread_bench.cu. Build and run both with
// ./devel/morton_spread/run_atomic_free_spread.sh.

#include "atomic_free_spread.cuh"

#include <cufinufft/spreadinterp.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace atomic_free_spread {
namespace {

// A failed allocation, memset, or copy in launch_spread would otherwise be
// reported only by the caller's cudaGetLastError(), which cannot see a failed
// cudaMemcpy of the active-region list. This prototype is benchmark-only, so a
// failure is fatal and loud instead of silently spreading into a partly
// initialised grid.
void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "atomic_free_spread: %s failed: %s\n", what,
                 cudaGetErrorString(status));
    std::abort();
  }
}

// A flattened Cartesian bin ID, with x varying fastest.
__global__ void assign_bins(const Point *points, int count, Grid grid, int *bin_ids) {
  using cufinufft::spreadinterp::fold_rescale;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  int bx = min(int(fold_rescale(points[i].x, grid.nx) / grid.bin_size), grid.bins_x - 1);
  int by = min(int(fold_rescale(points[i].y, grid.ny) / grid.bin_size), grid.bins_y - 1);
  int bz = min(int(fold_rescale(points[i].z, grid.nz) / grid.bin_size), grid.bins_z - 1);
  bin_ids[i] = bx + grid.bins_x * (by + grid.bins_y * bz);
}

// A source staged in shared memory for the window scan (and the narrow-grid tap
// fallback). One lane writes one record: the three integer support starts, the
// six kernel values of each dimension evaluated at that source's support start,
// and the strength.
struct StagedSource {
  int start[3];
  float kernel[3][WIDTH];
  float2 strength;
};

// One staging group per warp. 32 sources keeps every lane busy while staging;
// each record is 96 bytes with alignment, so a two-warp threadblock uses 6.1 KB
// of shared memory.
constexpr int STAGING_GROUP_SIZE = 32;
static_assert(STAGING_GROUP_SIZE <= 32);
static_assert(sizeof(StagedSource) == 96);

// Evaluate the six kernel values of one dimension and record where its support
// starts on the periodic fine grid. fold_rescale returns [0, grid_size), so the
// support start is in [-SUPPORT_OVERHANG, grid_size - SUPPORT_OVERHANG], and the
// Horner argument start_f - folded is in [-SUPPORT_OVERHANG,
// -SUPPORT_OVERHANG + 1], exactly the interval eval_kernel_vec_horner expects.
__device__ __forceinline__ void stage_axis(float *kernel, int &start,
                                           float coordinate, int grid_size) {
  using cufinufft::spreadinterp::eval_kernel_vec_horner;
  using cufinufft::spreadinterp::fold_rescale;
  const float folded = fold_rescale(coordinate, grid_size);
  const float start_f = ceilf(folded - (float)SUPPORT_OVERHANG);
  start = (int)start_f;
  eval_kernel_vec_horner<float, WIDTH>(kernel, start_f - folded, 2.0);
}

// Stage one source per lane. Lanes at or past count return from this helper
// only; the caller still executes its __syncwarp() with the whole warp.
__device__ __forceinline__ void stage_sources(
    StagedSource *__restrict__ staging, const Point *__restrict__ sources,
    int begin, int count, int lane, Grid grid) {
  if (lane >= count) return;
  const Point point = sources[begin + lane];
  StagedSource &staged = staging[lane];
  staged.strength = make_float2(point.strength.x, point.strength.y);
  stage_axis(staged.kernel[0], staged.start[0], point.x, grid.nx);
  stage_axis(staged.kernel[1], staged.start[1], point.y, grid.ny);
  stage_axis(staged.kernel[2], staged.start[2], point.z, grid.nz);
}

// Kernel value of one output cell in one dimension. start is the source's
// support start normalized to [0, grid_size) by the scan. A cell inside the grid
// is also in [0, grid_size), so cell - start is in (-grid_size, grid_size) and
// one conditional add folds it into [0, grid_size). A clipped edge cell
// (cell >= grid_size) has no output, and a tap outside [0, WIDTH) means the
// support does not reach the cell; both read as zero. This is correct for every
// grid size at least WIDTH, including the 6x6x6 case where a support wraps
// around the whole period.
__device__ __forceinline__ float tap_value(const float *kernel, int start,
                                           int cell, int grid_size) {
  int tap = cell - start;
  tap += (tap < 0) ? grid_size : 0;
  // A clipped edge cell has no output, and a tap outside the support is zero.
  // Both tests are selects so the kernel load stays predicated, not branched.
  return (cell < grid_size && tap < WIDTH) ? kernel[tap] : 0.0f;
}

// A source's support coverage of one lane-group axis. The support covers
// cell_count contiguous local cells starting at local_cell_offset; local cell
// local_cell_offset + k reads kernel[coefficient_offset + k].
struct SupportPlacement {
  int cell_count;
  int coefficient_offset;
  int local_cell_offset;
};

// Place one axis of a source support against a lane group, periodic in
// grid_size. The support start can be below the lane group when a source folds
// near zero and the lane group sits near the high grid edge; the conditional adds
// below fold that offset into one period. On a normal-width grid the covered
// cells are one contiguous run: either from local cell 0 through the high edge of
// the group, or from a local offset through the end of the group.
__device__ __forceinline__ SupportPlacement place_support(
    int support_start, int grid_size, int lane_group_start, int group_size) {
  int relative_support_start = support_start - lane_group_start;
  // The window paths only run on non-narrow dimensions (x/y >= 7, z >= 9), where
  // support_start - lane_group_start is never below -2 * grid_size. Two
  // conditional adds are therefore enough; they are branchless, unlike the
  // reference loop.
  relative_support_start += grid_size & (relative_support_start >> 31);
  relative_support_start += grid_size & (relative_support_start >> 31);

  // These results are selects, not branches.
  const bool below_group = relative_support_start < group_size;
  const bool wraps_high =
      relative_support_start >= grid_size - (WIDTH - 1);
  const int wrapped_cell_count =
      WIDTH - (grid_size - relative_support_start);
  SupportPlacement placement;
  placement.cell_count =
      below_group
          ? (group_size - relative_support_start)
          : (wraps_high ? min(wrapped_cell_count, group_size) : 0);
  placement.coefficient_offset =
      below_group ? 0
                  : (wraps_high ? (grid_size - relative_support_start) : 0);
  placement.local_cell_offset = below_group ? relative_support_start : 0;
  return placement;
}

// Non-wrapping fast path: plain interval intersection. Valid only when neither
// the support nor the lane group crosses a periodic grid boundary.
__device__ __forceinline__ SupportPlacement place_support_nowrap(
    int support_start, int lane_group_start, int group_size) {
  const int intersection_begin = max(support_start, lane_group_start);
  const int intersection_end =
      min(support_start + WIDTH, lane_group_start + group_size);
  SupportPlacement placement;
  placement.cell_count = max(intersection_end - intersection_begin, 0);
  placement.coefficient_offset = intersection_begin - support_start;
  placement.local_cell_offset = intersection_begin - lane_group_start;
  return placement;
}

// Kernel value at group-local cell `local` of a placement window. The window
// base is coefficient_offset - local_cell_offset, so cells outside the covered
// run fall before index 0 or past the six staged values and read as zero. The
// bounds check replaces the old padded coefficient rows.
__device__ __forceinline__ float window_value(const float *kernel, int base,
                                              int local) {
  const int index = base + local;
  return (index >= 0 && index < WIDTH) ? kernel[index] : 0.0f;
}

// The output region owned by one warp, clipped to the fine grid. nx, ny, nz are
// 1..WARP_REGION_* inside the grid and non-positive for a region past an edge.
struct WarpRegion {
  int x, y, z;       // global origin
  int nx, ny, nz;    // clipped extent
};

__device__ __forceinline__ WarpRegion make_warp_region(int x, int y, int z,
                                                       Grid grid) {
  return WarpRegion{x, y, z, min(WARP_REGION_X, grid.nx - x),
                    min(WARP_REGION_Y, grid.ny - y),
                    min(WARP_REGION_Z, grid.nz - z)};
}

// Decode a flat warp-region ID (x fastest) back to its clipped warp region. The
// spread, active-detection, and split-reduction kernels all use the same
// decoding, so the active list can carry bare region IDs.
__device__ __forceinline__ WarpRegion warp_region_from_id(int region_id, Grid grid,
                                                          int regions_x,
                                                          int regions_y) {
  const int region_z = region_id / (regions_x * regions_y);
  const int region_y = (region_id / regions_x) % regions_y;
  const int region_x = region_id % regions_x;
  return make_warp_region(region_x * WARP_REGION_X, region_y * WARP_REGION_Y,
                          region_z * WARP_REGION_Z, grid);
}

// The warp-region-local origin of the 2x2x4 cells written by one lane. Four
// lanes cover each x row, four rows cover y, and two z layers cover the 8x8x8
// warp region, so the 32 lane cell groups partition the region exactly once.
struct LaneCellGroup {
  int x, y, z;       // warp-region-local origin
};

__device__ __forceinline__ LaneCellGroup lane_cell_group(int lane) {
  const int lx = lane % LANES_X;
  const int ly = (lane / LANES_X) % LANES_Y;
  const int lz = lane / (LANES_X * LANES_Y);
  return LaneCellGroup{lx * LANE_CELLS_X, ly * LANE_CELLS_Y, lz * LANE_CELLS_Z};
}

// The 2x2x4 output cells accumulated by one lane.
struct CellAccumulator {
  float2 cell[LANE_CELLS_X][LANE_CELLS_Y][LANE_CELLS_Z];
};

// Add the 2x2x4 outer product of the three per-dimension kernel value sets,
// scaled by the source strength, to this lane's cell group. The 2 x 2 x-y
// products are reused across the four z cells and each z value is reused across
// the four x-y cells, which is the load reuse the lane cell group exists for.
__device__ __forceinline__ void accumulate_cell_group(
    CellAccumulator &acc, const float (&x_kernel_values)[LANE_CELLS_X],
    const float (&y_kernel_values)[LANE_CELLS_Y],
    const float (&z_kernel_values)[LANE_CELLS_Z], float2 strength) {
  float xy_weights[LANE_CELLS_X][LANE_CELLS_Y];
  #pragma unroll
  for (int lx = 0; lx < LANE_CELLS_X; ++lx)
    #pragma unroll
    for (int ly = 0; ly < LANE_CELLS_Y; ++ly)
      xy_weights[lx][ly] = x_kernel_values[lx] * y_kernel_values[ly];

  // Scale z by the source strength once per lane group, so each of the 16 cell
  // updates is a single FMA per component instead of a strength * xy * z
  // multiply.
  float2 strength_scaled_z[LANE_CELLS_Z];
  #pragma unroll
  for (int lz = 0; lz < LANE_CELLS_Z; ++lz)
    strength_scaled_z[lz] = make_float2(strength.x * z_kernel_values[lz],
                                        strength.y * z_kernel_values[lz]);

  #pragma unroll
  for (int lx = 0; lx < LANE_CELLS_X; ++lx)
    #pragma unroll
    for (int ly = 0; ly < LANE_CELLS_Y; ++ly)
      #pragma unroll
      for (int lz = 0; lz < LANE_CELLS_Z; ++lz) {
        acc.cell[lx][ly][lz].x += xy_weights[lx][ly] * strength_scaled_z[lz].x;
        acc.cell[lx][ly][lz].y += xy_weights[lx][ly] * strength_scaled_z[lz].y;
      }
}

// Shared per-source accumulation for the two window paths. The support covers a
// contiguous run of cells on each axis; if any axis covers none of this lane
// group, the whole 2x2x4 contribution is zero, so return before any shared load
// or FMA. This is the coverage early-out that the per-cell tap scan lacked.
__device__ __forceinline__ void accumulate_staged_source(
    CellAccumulator &acc, const StagedSource &staged, SupportPlacement x,
    SupportPlacement y, SupportPlacement z) {
  if (x.cell_count == 0 || y.cell_count == 0 || z.cell_count == 0) return;

  // Group cell c reads kernel[base + c]; window_value zeroes the cells outside
  // the covered run on either side.
  const int x_base = x.coefficient_offset - x.local_cell_offset;
  const int y_base = y.coefficient_offset - y.local_cell_offset;
  const int z_base = z.coefficient_offset - z.local_cell_offset;
  float x_kernel_values[LANE_CELLS_X];
  float y_kernel_values[LANE_CELLS_Y];
  float z_kernel_values[LANE_CELLS_Z];
  #pragma unroll
  for (int lx = 0; lx < LANE_CELLS_X; ++lx)
    x_kernel_values[lx] = window_value(staged.kernel[0], x_base, lx);
  #pragma unroll
  for (int ly = 0; ly < LANE_CELLS_Y; ++ly)
    y_kernel_values[ly] = window_value(staged.kernel[1], y_base, ly);
  #pragma unroll
  for (int lz = 0; lz < LANE_CELLS_Z; ++lz)
    z_kernel_values[lz] = window_value(staged.kernel[2], z_base, lz);

  accumulate_cell_group(acc, x_kernel_values, y_kernel_values,
                        z_kernel_values, staged.strength);
}

// General periodic window scan: three support placements against this lane
// group, then the covered kernel values.
__device__ __forceinline__ void scan_one_staged_source(
    const StagedSource &staged, Grid grid, int lane_group_x_start,
    int lane_group_y_start, int lane_group_z_start, CellAccumulator &acc) {
  const SupportPlacement x =
      place_support(staged.start[0], grid.nx, lane_group_x_start, LANE_CELLS_X);
  const SupportPlacement y =
      place_support(staged.start[1], grid.ny, lane_group_y_start, LANE_CELLS_Y);
  const SupportPlacement z =
      place_support(staged.start[2], grid.nz, lane_group_z_start, LANE_CELLS_Z);
  accumulate_staged_source(acc, staged, x, y, z);
}

// Non-wrapping window scan for a batch whose supports and warp region are all
// inside the grid. Interval intersection replaces the periodic placement.
__device__ __forceinline__ void scan_one_staged_source_nowrap(
    const StagedSource &staged, int lane_group_x_start, int lane_group_y_start,
    int lane_group_z_start, CellAccumulator &acc) {
  const SupportPlacement x =
      place_support_nowrap(staged.start[0], lane_group_x_start, LANE_CELLS_X);
  const SupportPlacement y =
      place_support_nowrap(staged.start[1], lane_group_y_start, LANE_CELLS_Y);
  const SupportPlacement z =
      place_support_nowrap(staged.start[2], lane_group_z_start, LANE_CELLS_Z);
  accumulate_staged_source(acc, staged, x, y, z);
}

// Hot path: general periodic support placement for every staged source.
__device__ __forceinline__ void scan_staging_group(
    CellAccumulator &acc, const StagedSource *__restrict__ staging, int count,
    Grid grid, int lane_group_x_start, int lane_group_y_start,
    int lane_group_z_start) {
  for (int j = 0; j < count; ++j)
    scan_one_staged_source(staging[j], grid, lane_group_x_start,
                           lane_group_y_start, lane_group_z_start, acc);
}

// Hot path for interior batches: non-wrapping support placement.
__device__ __forceinline__ void scan_staging_group_nowrap(
    CellAccumulator &acc, const StagedSource *__restrict__ staging, int count,
    int lane_group_x_start, int lane_group_y_start, int lane_group_z_start) {
  for (int j = 0; j < count; ++j)
    scan_one_staged_source_nowrap(staging[j], lane_group_x_start,
                                  lane_group_y_start, lane_group_z_start, acc);
}

// Narrow-grid fallback: per-cell periodic taps. A support can cover a lane-group
// prefix and suffix at the same time when grid_size <= WIDTH + LANE_CELLS - 2,
// which one contiguous placement window cannot express. These grids are tiny
// (e.g. 6x6x6), so this path is not hot.
__device__ __forceinline__ void scan_staging_group_tap(
    CellAccumulator &acc, const StagedSource *__restrict__ staging, int count,
    Grid grid, int lane_group_x_start, int lane_group_y_start,
    int lane_group_z_start) {
  for (int j = 0; j < count; ++j) {
    const StagedSource &staged = staging[j];
    // Normalize each support start into one period. The staged start is in
    // [-SUPPORT_OVERHANG, grid_size - SUPPORT_OVERHANG], so this also lets
    // tap_value fold with a single conditional add.
    const int start_x =
        staged.start[0] + ((staged.start[0] < 0) ? grid.nx : 0);
    const int start_y =
        staged.start[1] + ((staged.start[1] < 0) ? grid.ny : 0);
    const int start_z =
        staged.start[2] + ((staged.start[2] < 0) ? grid.nz : 0);
    float x_kernel_values[LANE_CELLS_X];
    float y_kernel_values[LANE_CELLS_Y];
    float z_kernel_values[LANE_CELLS_Z];
    #pragma unroll
    for (int lx = 0; lx < LANE_CELLS_X; ++lx)
      x_kernel_values[lx] = tap_value(staged.kernel[0], start_x,
                                      lane_group_x_start + lx, grid.nx);
    #pragma unroll
    for (int ly = 0; ly < LANE_CELLS_Y; ++ly)
      y_kernel_values[ly] = tap_value(staged.kernel[1], start_y,
                                      lane_group_y_start + ly, grid.ny);
    #pragma unroll
    for (int lz = 0; lz < LANE_CELLS_Z; ++lz)
      z_kernel_values[lz] = tap_value(staged.kernel[2], start_z,
                                      lane_group_z_start + lz, grid.nz);
    accumulate_cell_group(acc, x_kernel_values, y_kernel_values,
                          z_kernel_values, staged.strength);
  }
}

// Write this lane's cell group, skipping cells past the clipped warp-region
// extent. Cells outside the grid scan as zero weights and are never stored.
__device__ __forceinline__ void store_cell_group(
    const CellAccumulator &acc, cuFloatComplex *output, Grid grid,
    WarpRegion region, LaneCellGroup cell_group) {
  #pragma unroll
  for (int lx = 0; lx < LANE_CELLS_X; ++lx)
    #pragma unroll
    for (int ly = 0; ly < LANE_CELLS_Y; ++ly)
      #pragma unroll
      for (int lz = 0; lz < LANE_CELLS_Z; ++lz) {
        const int x = region.x + cell_group.x + lx;
        const int y = region.y + cell_group.y + ly;
        const int z = region.z + cell_group.z + lz;
        if (x < region.x + region.nx && y < region.y + region.ny &&
            z < region.z + region.nz) {
          const long long index =
              x + (long long)grid.nx * (y + (long long)grid.ny * z);
          output[index] = make_cuFloatComplex(acc.cell[lx][ly][lz].x,
                                              acc.cell[lx][ly][lz].y);
        }
      }
}

// Write this lane's cell group into one compact partial slice using warp-region-
// local cell indices. Only cells inside the clipped region extent are stored; the
// split reduction skips the remaining local cells and maps each surviving local
// cell to the global output index once.
__device__ __forceinline__ void store_cell_group_local(
    const CellAccumulator &acc, cuFloatComplex *region_slice, WarpRegion region,
    LaneCellGroup cell_group) {
  #pragma unroll
  for (int lx = 0; lx < LANE_CELLS_X; ++lx)
    #pragma unroll
    for (int ly = 0; ly < LANE_CELLS_Y; ++ly)
      #pragma unroll
      for (int lz = 0; lz < LANE_CELLS_Z; ++lz) {
        const int x = cell_group.x + lx;
        const int y = cell_group.y + ly;
        const int z = cell_group.z + lz;
        if (x < region.nx && y < region.ny && z < region.nz) {
          const int local_index =
              x + WARP_REGION_X * (y + WARP_REGION_Y * z);
          region_slice[local_index] =
              make_cuFloatComplex(acc.cell[lx][ly][lz].x,
                                  acc.cell[lx][ly][lz].y);
        }
      }
}

// Enumerate the non-empty candidate bins of one warp region, one contiguous
// sorted source range at a time. Every lane of a warp constructs and advances
// the same cursor, so the candidate-bin loops and __syncwarp() are warp-uniform.
class SourceRangeCursor {
 public:
  __device__ __forceinline__ SourceRangeCursor(WarpRegion region, Grid grid,
                                               const int *bin_offsets)
      : x_bins_(detail::candidate_bins(region.x, region.x + region.nx,
                                       grid.bin_size, grid.nx, grid.bins_x)),
        y_bins_(detail::candidate_bins(region.y, region.y + region.ny,
                                       grid.bin_size, grid.ny, grid.bins_y)),
        z_bins_(detail::candidate_bins(region.z, region.z + region.nz,
                                       grid.bin_size, grid.nz, grid.bins_z)),
        bin_offsets_(bin_offsets),
        grid_(grid) {}

  // Return the next non-empty candidate bin's source range. Candidate bins are
  // visited in z, y, x order; wrapped bins appear exactly once.
  __device__ __forceinline__ bool next(int &begin, int &end) {
    while (z_index_ < z_bins_.count) {
      const int bin = x_bins_[x_index_] +
                      grid_.bins_x *
                          (y_bins_[y_index_] + grid_.bins_y * z_bins_[z_index_]);
      begin = bin_offsets_[bin];
      end = bin_offsets_[bin + 1];
      if (++x_index_ == x_bins_.count) {
        x_index_ = 0;
        if (++y_index_ == y_bins_.count) {
          y_index_ = 0;
          ++z_index_;
        }
      }
      if (begin < end) return true;
    }
    return false;
  }

 private:
  detail::CandidateBins x_bins_;
  detail::CandidateBins y_bins_;
  detail::CandidateBins z_bins_;
  const int *bin_offsets_;
  Grid grid_;
  int x_index_ = 0;
  int y_index_ = 0;
  int z_index_ = 0;
};

// Sum the compact per-split partial slices into the output for active warp
// regions only. One task is one local cell of one active region: the split
// slices are summed in fixed order, and the local cell is mapped to the global
// output index once. Inactive regions are zeroed once by the host memset and are
// never read here. Each active-region cell of each split slice is fully defined
// by the spread kernel, which always stores its whole region.
__global__ void reduce_split_slices(const cuFloatComplex *__restrict__ partial,
                                    int split_count,
                                    const int *__restrict__ active_regions,
                                    int active_count, Grid grid,
                                    cuFloatComplex *__restrict__ output) {
  const int regions_x = warp_regions_x(grid);
  const int regions_y = warp_regions_y(grid);
  const long long task_count = (long long)active_count * WARP_REGION_CELLS;
  const float2 *const split_values = reinterpret_cast<const float2 *>(partial);
  for (long long task = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       task < task_count; task += (long long)blockDim.x * gridDim.x) {
    const int region_slot = (int)(task / WARP_REGION_CELLS);
    const int local_cell =
        (int)(task - (long long)region_slot * WARP_REGION_CELLS);
    const int local_x = local_cell % WARP_REGION_X;
    const int local_y = (local_cell / WARP_REGION_X) % WARP_REGION_Y;
    const int local_z = local_cell / (WARP_REGION_X * WARP_REGION_Y);
    const WarpRegion region = warp_region_from_id(active_regions[region_slot],
                                                  grid, regions_x, regions_y);
    if (local_x >= region.nx || local_y >= region.ny || local_z >= region.nz)
      continue;
    const int x = region.x + local_x;
    const int y = region.y + local_y;
    const int z = region.z + local_z;
    const long long index =
        x + (long long)grid.nx * (y + (long long)grid.ny * z);
    float2 value = split_values[(long long)region_slot * WARP_REGION_CELLS +
                                local_cell];
    for (int split = 1; split < split_count; ++split) {
      const float2 split_value =
          split_values[((long long)split * active_count + region_slot) *
                           WARP_REGION_CELLS +
                       local_cell];
      value.x += split_value.x;
      value.y += split_value.y;
    }
    output[index] = make_cuFloatComplex(value.x, value.y);
  }
}

// Scan and accumulate the sources of one warp region and store every cell of that
// region into region_output. region_output is either the final grid (single-split
// paths) or this split's slice of the partial grid (split path). The caller has
// already clipped the region; an out-of-grid region must not reach here.
//
// Each split scans a disjoint subset of every source range at staging-group
// granularity: it starts one staging group later per split index and then strides
// by split_count groups. split_count == 1 is exactly the single-warp loop, and a
// split with no sources still stores its whole region with zero accumulators, so
// every partial slice is fully defined for active regions.
__device__ __forceinline__ void spread_region(
    const Point *__restrict__ points, const int *__restrict__ bin_offsets,
    Grid grid, WarpRegion region, LaneCellGroup cell_group,
    cuFloatComplex *__restrict__ region_output, int split_index, int split_count,
    int lane, StagedSource *__restrict__ warp_staging) {
  const int lane_group_x_start = region.x + cell_group.x;
  const int lane_group_y_start = region.y + cell_group.y;
  const int lane_group_z_start = region.z + cell_group.z;
  // Only a fully interior warp region can use the branch-free support path. A
  // clipped region keeps lane groups past the grid edge, whose placement is
  // meaningless but never stored.
  const bool warp_region_inside_grid = region.nx == WARP_REGION_X &&
                                       region.ny == WARP_REGION_Y &&
                                       region.nz == WARP_REGION_Z;
  // A dimension at or below this width can wrap a support over both ends of the
  // lane group, which the contiguous window placement cannot express. This is a
  // whole-launch property, so it is loop-invariant and warp-uniform.
  const bool narrow_grid = grid.nx <= WIDTH + LANE_CELLS_X - 2 ||
                           grid.ny <= WIDTH + LANE_CELLS_Y - 2 ||
                           grid.nz <= WIDTH + LANE_CELLS_Z - 2;

  CellAccumulator acc = {};

  const int staging_group_stride = split_count * STAGING_GROUP_SIZE;
  SourceRangeCursor cursor(region, grid, bin_offsets);
  int begin, end;
  while (cursor.next(begin, end)) {
    for (int base = begin + split_index * STAGING_GROUP_SIZE; base < end;
         base += staging_group_stride) {
      const int count = min(STAGING_GROUP_SIZE, end - base);
      stage_sources(warp_staging, points, base, count, lane, grid);
      __syncwarp();
      // One warp-wide vote per staging group: if the region is interior and every
      // staged support is inside the grid, use non-wrapping placement for the
      // whole batch. Lanes past count vote true, and all lanes join the vote, so
      // the decision is uniform across the warp.
      bool support_inside_grid = true;
      if (lane < count) {
        const StagedSource &staged = warp_staging[lane];
        support_inside_grid =
            staged.start[0] >= 0 && staged.start[1] >= 0 &&
            staged.start[2] >= 0 &&
            staged.start[0] + WIDTH <= grid.nx &&
            staged.start[1] + WIDTH <= grid.ny &&
            staged.start[2] + WIDTH <= grid.nz;
      }
      const bool whole_group_nonwrapping =
          warp_region_inside_grid &&
          __all_sync(0xffffffffu, support_inside_grid);

      if (narrow_grid) {
        scan_staging_group_tap(acc, warp_staging, count, grid,
                               lane_group_x_start, lane_group_y_start,
                               lane_group_z_start);
      } else if (whole_group_nonwrapping) {
        scan_staging_group_nowrap(acc, warp_staging, count, lane_group_x_start,
                                  lane_group_y_start, lane_group_z_start);
      } else {
        scan_staging_group(acc, warp_staging, count, grid, lane_group_x_start,
                           lane_group_y_start, lane_group_z_start);
      }
      // Staging invariant: every lane finishes reading the single shared buffer
      // before the next staging group overwrites it.
      __syncwarp();
    }
  }

  // The split path writes a compact warp-region-local slice; the single-split
  // paths write global output indices directly.
  if (split_count > 1) {
    store_cell_group_local(acc, region_output, region, cell_group);
  } else {
    store_cell_group(acc, region_output, grid, region, cell_group);
  }
}

// Ask ptxas for at most about 102 registers per thread in the default two-warp
// configuration, which lets ten 64-thread threadblocks stay resident on the
// T1000. Other launch-policy values leave the register allocation to ptxas.
constexpr int MIN_THREADBLOCKS_PER_SM =
    (WARP_REGIONS_PER_THREADBLOCK == 2) ? 10 : 1;

// One warp executes one work item: a (warp region, split) pair. blockIdx.x is a
// block of WARPS_PER_THREADBLOCK consecutive work items, so a threadblock no
// longer corresponds to a contiguous tile; it is only a launch container.
__global__ __launch_bounds__(THREADS_PER_THREADBLOCK, MIN_THREADBLOCKS_PER_SM)
void spread(const Point *points, const int *bin_offsets, Grid grid,
            cuFloatComplex *output, const int *active_regions, int split_count,
            int active_count, int work_count) {
  // One single-buffered staging group per warp. Each work item owns one warp
  // region, so the buffer is warp-private and needs only __syncwarp().
  __shared__ __align__(16) StagedSource staging[WARPS_PER_THREADBLOCK]
                                                 [STAGING_GROUP_SIZE];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;

  const int work_id = int(blockIdx.x) * WARPS_PER_THREADBLOCK + warp;
  if (work_id >= work_count) return;
  const int region_slot = work_id / split_count;
  const int split_index = work_id - region_slot * split_count;
  // The full-grid fast path passes no active list and indexes flat warp-region
  // IDs directly; the active-list paths decode the compacted region list.
  const int region_id =
      (active_regions != nullptr) ? active_regions[region_slot] : work_id;
  const WarpRegion region = warp_region_from_id(region_id, grid, warp_regions_x(grid),
                                                warp_regions_y(grid));
  // This branch is warp-uniform: all lanes of a warp see the same region, so the
  // early return never leaves part of a warp outside a later __syncwarp(). The
  // other warps in the threadblock are independent because the scan has no
  // threadblock-wide barrier.
  if (region.nx <= 0 || region.ny <= 0 || region.nz <= 0) return;

  // A split writes its whole warp region into its compact partial slice; the
  // single-split paths write the final grid directly.
  cuFloatComplex *const region_output =
      (split_count > 1)
          ? output + ((long long)split_index * active_count + region_slot) *
                         WARP_REGION_CELLS
          : output;

  spread_region(points, bin_offsets, grid, region, lane_cell_group(lane),
                region_output, split_index, split_count, lane,
                &staging[warp][0]);
}

}  // namespace

ActiveRegionPlan plan_active_regions(const int *host_bin_offsets, Grid grid) {
  const int regions_x = warp_regions_x(grid);
  const int regions_y = warp_regions_y(grid);
  const int regions_z = warp_regions_z(grid);
  const int region_count = regions_x * regions_y * regions_z;

  ActiveRegionPlan plan;
  plan.regions.reserve((size_t)region_count);

  // A region is active when the sum of the source counts of its candidate bins
  // is nonzero. The candidate-bin set is exactly the one SourceRangeCursor
  // enumerates, so the test can only over-report: a source in a candidate bin
  // may still miss the region, but no region that would accumulate a
  // contribution is ever skipped.
  for (int region_z = 0; region_z < regions_z; ++region_z) {
    const int z = region_z * WARP_REGION_Z;
    const int remaining_z = grid.nz - z;
    const int nz = remaining_z < WARP_REGION_Z ? remaining_z : WARP_REGION_Z;
    if (nz <= 0) continue;
    for (int region_y = 0; region_y < regions_y; ++region_y) {
      const int y = region_y * WARP_REGION_Y;
      const int remaining_y = grid.ny - y;
      const int ny = remaining_y < WARP_REGION_Y ? remaining_y : WARP_REGION_Y;
      if (ny <= 0) continue;
      for (int region_x = 0; region_x < regions_x; ++region_x) {
        const int x = region_x * WARP_REGION_X;
        const int remaining_x = grid.nx - x;
        const int nx =
            remaining_x < WARP_REGION_X ? remaining_x : WARP_REGION_X;
        if (nx <= 0) continue;

        const detail::CandidateBins x_bins = detail::candidate_bins(
            x, x + nx, grid.bin_size, grid.nx, grid.bins_x);
        const detail::CandidateBins y_bins = detail::candidate_bins(
            y, y + ny, grid.bin_size, grid.ny, grid.bins_y);
        const detail::CandidateBins z_bins = detail::candidate_bins(
            z, z + nz, grid.bin_size, grid.nz, grid.bins_z);
        int source_count = 0;
        for (int z_index = 0; z_index < z_bins.count; ++z_index)
          for (int y_index = 0; y_index < y_bins.count; ++y_index)
            for (int x_index = 0; x_index < x_bins.count; ++x_index) {
              const int bin =
                  x_bins[x_index] +
                  grid.bins_x *
                      (y_bins[y_index] + grid.bins_y * z_bins[z_index]);
              source_count +=
                  host_bin_offsets[bin + 1] - host_bin_offsets[bin];
            }
        if (source_count != 0) {
          const int region_id =
              region_x + regions_x * (region_y + regions_y * region_z);
          plan.regions.push_back(region_id);
        }
      }
    }
  }

  const int active_count = (int)plan.regions.size();
  if (active_count == 0) {
    plan.split_count = 1;
    plan.path = LaunchPath::Empty;
    return plan;
  }

  // Split heuristic: aim for about 2048 work items when few regions are active
  // (clustered and dense inputs), otherwise 512, and never exceed 32. Uniform
  // and sparse inputs stay at split_count == 1.
  const long long split_target = active_count < 512 ? 2048 : 512;
  plan.split_count = 1;
  while (plan.split_count < 32 &&
         (long long)active_count * plan.split_count < split_target)
    plan.split_count <<= 1;

  if (plan.split_count == 1 && active_count == region_count)
    plan.path = LaunchPath::FullGrid;
  else if (plan.split_count == 1)
    plan.path = LaunchPath::ActiveSingleSplit;
  else
    plan.path = LaunchPath::ActiveSplit;
  return plan;
}

void launch_assign_bins(const Point *points, int count, Grid grid, int *bin_ids) {
  constexpr int THREADS = 256;
  assign_bins<<<(count + THREADS - 1) / THREADS, THREADS>>>(points, count, grid, bin_ids);
}

void launch_spread(const Point *points, const int *device_bin_offsets,
                   const int *host_bin_offsets, Grid grid,
                   cuFloatComplex *output) {
  const int region_count = warp_region_count(grid);
  const long long cells = cell_count(grid);
  const size_t output_bytes = (size_t)cells * sizeof(cuFloatComplex);

  // The active-region plan is recomputed here so the caller's spread interval
  // includes the host scan, the active-list upload, the kernel, the output memset,
  // and the split reduction. Temporaries are allocated per launch with
  // cudaMalloc/cudaFree; there is no persistent workspace yet.
  const ActiveRegionPlan plan = plan_active_regions(host_bin_offsets, grid);
  const int active_count = (int)plan.regions.size();
  const int split_count = plan.split_count;

  constexpr int REDUCE_THREADS = 256;
  const dim3 threads(THREADS_PER_THREADBLOCK);

  if (active_count == 0) {
    // No warp region has a source: every cell is a zero.
    check_cuda(cudaMemsetAsync(output, 0, output_bytes),
               "cudaMemsetAsync(output)");
    return;
  }

  int *active_regions = nullptr;
  cuFloatComplex *partial = nullptr;

  if (plan.path != LaunchPath::FullGrid) {
    check_cuda(cudaMalloc(&active_regions, (size_t)active_count * sizeof(int)),
               "cudaMalloc(active_regions)");
    check_cuda(cudaMemcpy(active_regions, plan.regions.data(),
                          (size_t)active_count * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy(active_regions)");
  }

  if (plan.path == LaunchPath::ActiveSplit) {
    check_cuda(cudaMalloc(&partial, (size_t)split_count * (size_t)active_count *
                                        WARP_REGION_CELLS *
                                        sizeof(cuFloatComplex)),
               "cudaMalloc(partial)");
  }

  if (plan.path == LaunchPath::FullGrid) {
    // Full-grid fast path: every region is active and there is one split, so the
    // active list is only an identity map. Write the output directly, no memset.
    spread<<<(region_count + WARPS_PER_THREADBLOCK - 1) /
                 WARPS_PER_THREADBLOCK,
             threads>>>(points, device_bin_offsets, grid, output, nullptr, 1,
                        active_count, region_count);
  } else if (plan.path == LaunchPath::ActiveSingleSplit) {
    // Active-list single-split path: skip empty regions, but zero the output first
    // because those regions are never written.
    check_cuda(cudaMemsetAsync(output, 0, output_bytes),
               "cudaMemsetAsync(output)");
    spread<<<(active_count + WARPS_PER_THREADBLOCK - 1) /
                 WARPS_PER_THREADBLOCK,
             threads>>>(points, device_bin_offsets, grid, output, active_regions,
                        1, active_count, active_count);
  } else {
    // Split path: each work item writes its whole warp region into its compact
    // partial slice, so every active-region cell of every slice is defined.
    const int work_count = active_count * split_count;
    spread<<<(work_count + WARPS_PER_THREADBLOCK - 1) /
                 WARPS_PER_THREADBLOCK,
             threads>>>(points, device_bin_offsets, grid, partial,
                        active_regions, split_count, active_count, work_count);
    // Zero the inactive regions once; the reduction writes only active cells.
    check_cuda(cudaMemsetAsync(output, 0, output_bytes),
               "cudaMemsetAsync(output)");
    const long long reduce_tasks =
        (long long)active_count * WARP_REGION_CELLS;
    const long long reduce_blocks =
        (reduce_tasks + REDUCE_THREADS - 1) / REDUCE_THREADS;
    reduce_split_slices<<<(unsigned int)reduce_blocks, REDUCE_THREADS>>>(
        partial, split_count, active_regions, active_count, grid, output);
    check_cuda(cudaFree(partial), "cudaFree(partial)");
  }

  if (active_regions != nullptr)
    check_cuda(cudaFree(active_regions), "cudaFree(active_regions)");
}

}  // namespace atomic_free_spread
