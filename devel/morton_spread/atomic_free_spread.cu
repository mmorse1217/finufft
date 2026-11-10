// Atomic-free 3D type-1 spreading prototype: implementation.
//
// One warp owns an 8x8x8 warp region and each lane owns a 2x2x4 lane cell
// group, so each output cell is written by exactly one lane. Sources are staged
// per warp as six Horner kernel values per dimension plus the support start.
// The scan places each support against the lane group and reuses the covered
// 2/2/4 kernel values. There are no spreading atomics.
//
// Every launch uses the same path: the host builds a compact list of active warp
// regions, each work item scans one (region, split) pair into a compact partial
// slice, and a reduction sums the splits into the output. A full-grid input is
// simply a list containing every region with split_count == 1.
//
// Binning and sorting are done on the CPU after a GPU kernel computes the bin
// IDs, so bin assignment uses the same GPU fold_rescale as spreading. This is a
// correctness/readability prototype, not an optimized implementation.
//
// The entry point, timing, reference comparison, and correctness checks are in
// atomic_free_spread_bench.cu. Build and run both with
// ./devel/morton_spread/run_atomic_free_spread.sh.

#include "atomic_free_spread.cuh"

#include <cufinufft/spreadinterp.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace atomic_free_spread {
namespace {

void check_cuda(cudaError_t status, const char *what) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "atomic_free_spread: %s failed: %s\n", what,
                 cudaGetErrorString(status));
    std::abort();
  }
}

__global__ void assign_bins(const Point *points, int count, Grid grid, int *bin_ids) {
  using cufinufft::spreadinterp::fold_rescale;
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int bx =
      min(int(fold_rescale(points[i].x, grid.nx) / grid.bin_size), grid.bins_x - 1);
  const int by =
      min(int(fold_rescale(points[i].y, grid.ny) / grid.bin_size), grid.bins_y - 1);
  const int bz =
      min(int(fold_rescale(points[i].z, grid.nz) / grid.bin_size), grid.bins_z - 1);
  bin_ids[i] = bx + grid.bins_x * (by + grid.bins_y * bz);
}

// One source staged by one lane. kernel[d][k] is the k-th kernel value along
// dimension d, evaluated at the source's support start.
struct StagedSource {
  int start[3];
  float kernel[3][WIDTH];
  float2 strength;
};

constexpr int STAGING_GROUP_SIZE = 32;
static_assert(STAGING_GROUP_SIZE <= 32);
static_assert(sizeof(StagedSource) == 96);

__device__ __forceinline__ void stage_axis(float *kernel, int &start,
                                           float coordinate, int grid_size) {
  using cufinufft::spreadinterp::eval_kernel_vec_horner;
  using cufinufft::spreadinterp::fold_rescale;
  const float folded = fold_rescale(coordinate, grid_size);
  const float start_f = ceilf(folded - (float)SUPPORT_OVERHANG);
  start = (int)start_f;
  eval_kernel_vec_horner<float, WIDTH>(kernel, start_f - folded, 2.0);
}

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

// A support placement on one axis: cell_count contiguous lane-group cells
// starting at local_cell_offset, reading kernel[coefficient_offset + k].
struct SupportPlacement {
  int cell_count;
  int coefficient_offset;
  int local_cell_offset;
};

// Contiguous support placement for grids wide enough that a support cannot
// cover both ends of a lane group. The support start may be below the group by
// at most one period.
__device__ __forceinline__ SupportPlacement place_support(
    int support_start, int grid_size, int lane_group_start, int group_size) {
  int relative_support_start = support_start - lane_group_start;
  relative_support_start += grid_size & (relative_support_start >> 31);
  relative_support_start += grid_size & (relative_support_start >> 31);

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

// Plain interval intersection for supports and groups that do not wrap.
__device__ __forceinline__ SupportPlacement place_support_nowrap(
    int support_start, int lane_group_start, int group_size) {
  const int begin = max(support_start, lane_group_start);
  const int end = min(support_start + WIDTH, lane_group_start + group_size);
  SupportPlacement placement;
  placement.cell_count = max(end - begin, 0);
  placement.coefficient_offset = begin - support_start;
  placement.local_cell_offset = begin - lane_group_start;
  return placement;
}

__device__ __forceinline__ float window_value(const float *kernel, int base,
                                              int local) {
  const int index = base + local;
  return (index >= 0 && index < WIDTH) ? kernel[index] : 0.0f;
}

__device__ __forceinline__ float tap_value(const float *kernel, int start,
                                           int cell, int grid_size) {
  int tap = cell - start;
  tap += (tap < 0) ? grid_size : 0;
  return (cell < grid_size && tap < WIDTH) ? kernel[tap] : 0.0f;
}

// The output region owned by one warp, clipped to the fine grid.
struct WarpRegion {
  int x, y, z;
  int nx, ny, nz;
};

__device__ __forceinline__ WarpRegion make_warp_region(int x, int y, int z,
                                                       Grid grid) {
  return WarpRegion{x, y, z, min(WARP_REGION_X, grid.nx - x),
                    min(WARP_REGION_Y, grid.ny - y),
                    min(WARP_REGION_Z, grid.nz - z)};
}

__device__ __forceinline__ WarpRegion warp_region_from_id(int region_id, Grid grid,
                                                          int regions_x,
                                                          int regions_y) {
  const int region_z = region_id / (regions_x * regions_y);
  const int region_y = (region_id / regions_x) % regions_y;
  const int region_x = region_id % regions_x;
  return make_warp_region(region_x * WARP_REGION_X, region_y * WARP_REGION_Y,
                          region_z * WARP_REGION_Z, grid);
}

// The warp-region-local origin of the 2x2x4 cells written by one lane.
struct LaneCellGroup {
  int x, y, z;
};

__device__ __forceinline__ LaneCellGroup lane_cell_group(int lane) {
  const int lx = lane % LANES_X;
  const int ly = (lane / LANES_X) % LANES_Y;
  const int lz = lane / (LANES_X * LANES_Y);
  return LaneCellGroup{lx * LANE_CELLS_X, ly * LANE_CELLS_Y, lz * LANE_CELLS_Z};
}

struct CellAccumulator {
  float2 cell[LANE_CELLS_X][LANE_CELLS_Y][LANE_CELLS_Z];
};

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

__device__ __forceinline__ void accumulate_staged_source(
    CellAccumulator &acc, const StagedSource &staged, SupportPlacement x,
    SupportPlacement y, SupportPlacement z) {
  if (x.cell_count == 0 || y.cell_count == 0 || z.cell_count == 0) return;

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

__device__ __forceinline__ void scan_staging_group(
    CellAccumulator &acc, const StagedSource *__restrict__ staging, int count,
    Grid grid, int lane_group_x_start, int lane_group_y_start,
    int lane_group_z_start) {
  for (int j = 0; j < count; ++j) {
    const StagedSource &staged = staging[j];
    accumulate_staged_source(
        acc, staged,
        place_support(staged.start[0], grid.nx, lane_group_x_start,
                      LANE_CELLS_X),
        place_support(staged.start[1], grid.ny, lane_group_y_start,
                      LANE_CELLS_Y),
        place_support(staged.start[2], grid.nz, lane_group_z_start,
                      LANE_CELLS_Z));
  }
}

__device__ __forceinline__ void scan_staging_group_nowrap(
    CellAccumulator &acc, const StagedSource *__restrict__ staging, int count,
    int lane_group_x_start, int lane_group_y_start, int lane_group_z_start) {
  for (int j = 0; j < count; ++j) {
    const StagedSource &staged = staging[j];
    accumulate_staged_source(
        acc, staged,
        place_support_nowrap(staged.start[0], lane_group_x_start,
                             LANE_CELLS_X),
        place_support_nowrap(staged.start[1], lane_group_y_start,
                             LANE_CELLS_Y),
        place_support_nowrap(staged.start[2], lane_group_z_start,
                             LANE_CELLS_Z));
  }
}

// Correctness fallback for grids narrow enough that one support can cover both
// a prefix and a suffix of a lane group. These grids are tiny, so the per-cell
// taps are not hot.
__device__ __forceinline__ void scan_staging_group_tap(
    CellAccumulator &acc, const StagedSource *__restrict__ staging, int count,
    Grid grid, int lane_group_x_start, int lane_group_y_start,
    int lane_group_z_start) {
  for (int j = 0; j < count; ++j) {
    const StagedSource &staged = staging[j];
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

// Enumerate the non-empty candidate bins of one warp region as contiguous
// sorted source ranges. All lanes of the warp advance the same cursor.
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

// Store this lane's cell group into one warp-region-local partial slice.
__device__ __forceinline__ void store_region_slice(
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

// Sum all split slices of each active region into the output. Inactive regions
// are zeroed once by the host and are not visited here.
__global__ void reduce_partial(const cuFloatComplex *__restrict__ partial,
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

    const long long region_base =
        (long long)region_slot * split_count * WARP_REGION_CELLS + local_cell;
    float2 value = split_values[region_base];
    for (int split = 1; split < split_count; ++split) {
      const float2 split_value =
          split_values[region_base + (long long)split * WARP_REGION_CELLS];
      value.x += split_value.x;
      value.y += split_value.y;
    }

    const int x = region.x + local_x;
    const int y = region.y + local_y;
    const int z = region.z + local_z;
    const long long index =
        x + (long long)grid.nx * (y + (long long)grid.ny * z);
    output[index] = make_cuFloatComplex(value.x, value.y);
  }
}

// Split `split_index` scans staging groups split_index, split_index +
// split_count, ... of every source range. split_count == 1 scans the range
// normally; an empty split still stores a zero slice so every partial slice is
// fully defined.
__device__ __forceinline__ void spread_region(
    const Point *__restrict__ points, const int *__restrict__ bin_offsets,
    Grid grid, WarpRegion region, LaneCellGroup cell_group,
    cuFloatComplex *__restrict__ region_slice, int split_index, int split_count,
    int lane, StagedSource *__restrict__ warp_staging) {
  const int lane_group_x_start = region.x + cell_group.x;
  const int lane_group_y_start = region.y + cell_group.y;
  const int lane_group_z_start = region.z + cell_group.z;
  const bool region_inside_grid = region.nx == WARP_REGION_X &&
                                  region.ny == WARP_REGION_Y &&
                                  region.nz == WARP_REGION_Z;
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
          region_inside_grid &&
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
      __syncwarp();
    }
  }

  store_region_slice(acc, region_slice, region, cell_group);
}

constexpr int MIN_THREADBLOCKS_PER_SM =
    (WARP_REGIONS_PER_THREADBLOCK == 2) ? 10 : 1;

// One warp owns one work item: a (warp region, split) pair.
__global__ __launch_bounds__(THREADS_PER_THREADBLOCK, MIN_THREADBLOCKS_PER_SM)
void spread(const Point *points, const int *bin_offsets, Grid grid,
            cuFloatComplex *partial, const int *active_regions,
            int split_count, int work_count) {
  __shared__ __align__(16) StagedSource staging[WARPS_PER_THREADBLOCK]
                                                 [STAGING_GROUP_SIZE];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int work_id = int(blockIdx.x) * WARPS_PER_THREADBLOCK + warp;
  if (work_id >= work_count) return;

  const int region_slot = work_id / split_count;
  const int split_index = work_id - region_slot * split_count;
  const WarpRegion region = warp_region_from_id(
      active_regions[region_slot], grid, warp_regions_x(grid),
      warp_regions_y(grid));
  if (region.nx <= 0 || region.ny <= 0 || region.nz <= 0) return;

  cuFloatComplex *const region_slice =
      partial +
      ((long long)region_slot * split_count + split_index) * WARP_REGION_CELLS;
  spread_region(points, bin_offsets, grid, region, lane_cell_group(lane),
                region_slice, split_index, split_count, lane,
                &staging[warp][0]);
}

// Split target: keep enough work items to fill several waves on a small GPU.
// A region-rich input stays at one split; only inputs with few active regions
// are split, up to a fixed cap.
constexpr int SPLIT_TARGET_WORK_ITEMS = 2048;
constexpr int MAX_SPLIT_COUNT = 32;

struct RegionPlan {
  std::vector<int> regions;
  int split_count = 1;
};

RegionPlan plan_active_regions(const int *host_bin_offsets, Grid grid) {
  const int regions_x = warp_regions_x(grid);
  const int regions_y = warp_regions_y(grid);
  const int regions_z = warp_regions_z(grid);

  RegionPlan plan;
  plan.regions.reserve((size_t)regions_x * regions_y * regions_z);

  // A region is active when any candidate bin contains a source. The candidate
  // set matches SourceRangeCursor, so this can over-report a region but never
  // skip one that would receive a contribution.
  for (int region_z = 0; region_z < regions_z; ++region_z) {
    const int z = region_z * WARP_REGION_Z;
    const int nz = min(WARP_REGION_Z, grid.nz - z);
    if (nz <= 0) continue;
    for (int region_y = 0; region_y < regions_y; ++region_y) {
      const int y = region_y * WARP_REGION_Y;
      const int ny = min(WARP_REGION_Y, grid.ny - y);
      if (ny <= 0) continue;
      for (int region_x = 0; region_x < regions_x; ++region_x) {
        const int x = region_x * WARP_REGION_X;
        const int nx = min(WARP_REGION_X, grid.nx - x);
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

        if (source_count != 0)
          plan.regions.push_back(
              region_x + regions_x * (region_y + regions_y * region_z));
      }
    }
  }

  const int active_count = (int)plan.regions.size();
  while (plan.split_count < MAX_SPLIT_COUNT &&
         (long long)active_count * plan.split_count <
             SPLIT_TARGET_WORK_ITEMS)
    plan.split_count <<= 1;
  return plan;
}

}  // namespace

void launch_assign_bins(const Point *points, int count, Grid grid, int *bin_ids) {
  constexpr int THREADS = 256;
  assign_bins<<<(count + THREADS - 1) / THREADS, THREADS>>>(points, count, grid,
                                                            bin_ids);
}

void launch_spread(const Point *points, const int *device_bin_offsets,
                   const int *host_bin_offsets, Grid grid,
                   cuFloatComplex *output) {
  const int region_count = warp_region_count(grid);
  const size_t output_bytes =
      (size_t)cell_count(grid) * sizeof(cuFloatComplex);

  // The plan stays inside the timed launch: the caller's spread interval
  // includes plan, upload, kernel, memset, and reduction.
  const RegionPlan plan = plan_active_regions(host_bin_offsets, grid);
  const int active_count = (int)plan.regions.size();
  if (active_count == 0) {
    check_cuda(cudaMemsetAsync(output, 0, output_bytes),
               "cudaMemsetAsync(output)");
    return;
  }

  const int split_count = plan.split_count;
  const int work_count = active_count * split_count;
  const size_t partial_bytes =
      (size_t)work_count * WARP_REGION_CELLS * sizeof(cuFloatComplex);

  // One allocation holds both the active list and the compact partial grid.
  void *workspace = nullptr;
  check_cuda(cudaMalloc(&workspace, partial_bytes +
                                       (size_t)active_count * sizeof(int)),
             "cudaMalloc(workspace)");
  cuFloatComplex *const partial = static_cast<cuFloatComplex *>(workspace);
  int *const active_regions = reinterpret_cast<int *>(
      static_cast<char *>(workspace) + partial_bytes);
  check_cuda(cudaMemcpy(active_regions, plan.regions.data(),
                        (size_t)active_count * sizeof(int),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy(active_regions)");

  spread<<<(work_count + WARPS_PER_THREADBLOCK - 1) /
               WARPS_PER_THREADBLOCK,
           dim3(THREADS_PER_THREADBLOCK)>>>(points, device_bin_offsets, grid,
                                            partial, active_regions, split_count,
                                            work_count);

  // The reduction writes every cell of every active region. Only cells of
  // inactive regions need zeroing.
  if (active_count != region_count)
    check_cuda(cudaMemsetAsync(output, 0, output_bytes),
               "cudaMemsetAsync(output)");

  constexpr int REDUCE_THREADS = 256;
  const long long reduce_tasks =
      (long long)active_count * WARP_REGION_CELLS;
  const unsigned int reduce_blocks =
      (unsigned int)((reduce_tasks + REDUCE_THREADS - 1) / REDUCE_THREADS);
  reduce_partial<<<reduce_blocks, REDUCE_THREADS>>>(
      partial, split_count, active_regions, active_count, grid, output);

  check_cuda(cudaFree(workspace), "cudaFree(workspace)");
}

}  // namespace atomic_free_spread
