// Readable, atomic-free 3D type-1 spreading prototype: implementation.
//
// This is a standalone experiment, not part of the library build. It uses a
// width-6 kernel (sigma=2, tolerance 1e-5), single precision, and periodic
// boundaries. One warp owns one 8x8x8 warp region, and each lane owns a 2x2x4
// lane cell group, so each output cell is written exactly once by one lane.
// There are no spreading atomics.
//
// The launch is split into four phases: detect the active warp regions (any
// non-empty candidate bin), compact them into a deterministic list, launch one
// work item per (active region, split) pair, and (when splits are used) reduce
// the per-split partial grids into the output. A spread kernel launch therefore
// carries a work-item count instead of a threadblock tile grid: a full-grid
// launch passes no active list and indexes flat warp-region IDs directly, while
// an active-list launch decodes active_regions[work_id / split_count] and takes
// split_index = work_id % split_count. Inactive regions are zeroed by the host
// memset and are never touched by the kernel or the reduction.
//
// Sources are staged in shared memory in exp_a's 30-float record form: three
// support starts, the strength, and six Horner kernel coefficients per
// dimension. Nine lanes cooperate on one source, so three sources share one
// warp iteration, and each lane then places the support against its own 2x2x4
// cell group with min/max interval arithmetic instead of per-cell weights.
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
// cudaMemcpy of the active-region count. This prototype is benchmark-only, so a
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

// The nine polynomial coefficients of one kernel value, one row per Horner term
// and one column per kernel value index 0..WIDTH-1. Copied verbatim from
// exp_a_cartesian.cu: the kernel is evaluated as a degree-8 polynomial in the
// Horner argument 2 * (support_start - folded) + (WIDTH - 1), which is
// eval_kernel_vec_horner()'s exact fold + Horner scheme, so the spread matches
// cuFINUFFT bit-for-bit per source. Nine staging lanes of one dimension load
// three columns from constant memory into registers, and the same table is
// copied to shared memory so the second coefficient set (index + 3) can be
// evaluated with a dynamic column index without a constant-memory replay.
__device__ __constant__ float KERNEL_COEFFICIENT_TABLE[9][WIDTH] = {
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

// Sources staged together in one shared-memory batch, and the float slots each
// staged source occupies. Nine staging lanes cover a source's 18 coefficients
// (two per lane), so three sources share one warp iteration and 13 leaves
// three-point iterations with one or two points in the last one.
constexpr int MAX_POINTS_PER_BATCH = 13;
static_assert(MAX_POINTS_PER_BATCH <= 32);

// 30-float staged-source layout, copied from exp_a_cartesian.cu:
//   0..2    support starts x, y, z (int bits, rewritten in place by
//           prepare_two_coefficients; the raw pass leaves folded coordinates
//           here first)
//   3..4    strength real, imag
//   5..10   six x kernel coefficients, index = support coefficient offset
//   11      zero pad: the x post-pad doubles as the y pre-pad
//   12..17  six y kernel coefficients
//   18..20  zero pads: the z pre-pads
//   21..26  six z kernel coefficients
//   27..29  zero pads: the z post-pads
// The pads let the scan load a lane group's coefficients with one dynamic base
// and no range selects: x needs one post-pad plus two selects, y needs only
// slots 11 and 18 as pads, and z reads four contiguous slots from base
// 21 + offset - local_cell_offset in [18, 26].
constexpr int VALUES_PER_POINT = 30;
constexpr int SUPPORT_START_SLOT = 0;   // 0..2
constexpr int STRENGTH_SLOT = 3;        // 3..4
constexpr int X_COEFFICIENT_SLOT = 5;   // 5..10
constexpr int X_Y_PAD_SLOT = 11;
constexpr int Y_COEFFICIENT_SLOT = 12;  // 12..17
constexpr int Z_COEFFICIENT_SLOT = 21;  // 21..26
constexpr int Z_PADDING_BEFORE = 3;     // 18..20
constexpr int Z_PADDING_AFTER = 3;      // 27..29
static_assert(X_Y_PAD_SLOT == X_COEFFICIENT_SLOT + WIDTH);
static_assert(Y_COEFFICIENT_SLOT == X_Y_PAD_SLOT + 1);
static_assert(Z_COEFFICIENT_SLOT == Y_COEFFICIENT_SLOT + WIDTH + Z_PADDING_BEFORE);
static_assert(VALUES_PER_POINT == Z_COEFFICIENT_SLOT + WIDTH + Z_PADDING_AFTER);
static_assert((VALUES_PER_POINT * int(sizeof(float))) % 8 == 0);  // int2 metadata load

// Raw pass for one batch: lanes 0..count-1 issue all the global loads of the
// batch at once, so the warp waits on one round of loads instead of `count`
// serialized ones. Coordinates and strength land in slots 0..4; the coefficient
// pass fills in the support starts and coefficients afterwards. Point is AoS
// here (exp_a keeps four parallel arrays), so each lane's five floats are a
// 20-byte strided gather; a 13-source batch still spans only a few 128-byte
// lines of the sorted point table.
__device__ __forceinline__ void stage_raw_points(const Point *__restrict__ sources,
                                                int first_point_index,
                                                int staged_point_count, int lane,
                                                float *__restrict__ staging) {
  if (lane < staged_point_count) {
    const Point point = sources[first_point_index + lane];
    float *const staged_point = staging + lane * VALUES_PER_POINT;
    staged_point[SUPPORT_START_SLOT] = point.x;
    staged_point[SUPPORT_START_SLOT + 1] = point.y;
    staged_point[SUPPORT_START_SLOT + 2] = point.z;
    staged_point[STRENGTH_SLOT] = point.strength.x;
    staged_point[STRENGTH_SLOT + 1] = point.strength.y;
  }
}

// Two coefficients for one staged source/dimension. The primary coefficient
// index k comes from the register-resident nine-term set, and k+3 from the
// shared copy, so nine lanes cover a source's 18 coefficients and three sources
// share one warp iteration (lanes 27..31 are idle). Each lane folds the same
// broadcast coordinate and rewrites the support start slot with the same int
// value, which is a benign write race, not a divergence.
__device__ __forceinline__ void prepare_two_coefficients(
    float *__restrict__ staged_point, int grid_size_dimension,
    int support_start_slot, int coefficient_slot, int second_coefficient_index,
    const float *__restrict__ kernel_coefficients,
    const float (*__restrict__ second_kernel_coefficients)[WIDTH]) {
  using cufinufft::spreadinterp::fold_rescale;
  const float folded_coordinate =
      fold_rescale(staged_point[support_start_slot], grid_size_dimension);
  const float support_start_float =
      ceilf(folded_coordinate - (float)SUPPORT_OVERHANG);
  const int support_start_int = (int)support_start_float;
  const float horner_argument =
      __fmaf_rn(2.f, support_start_float - folded_coordinate, (float)(WIDTH - 1));
  reinterpret_cast<int *>(staged_point)[support_start_slot] = support_start_int;

  float coefficient_value = kernel_coefficients[8];
  #pragma unroll
  for (int horner_term = 7; horner_term >= 0; horner_term--)
    coefficient_value = __fmaf_rn(horner_argument, coefficient_value,
                                  kernel_coefficients[horner_term]);
  staged_point[coefficient_slot] = coefficient_value;

  float second_coefficient_value = second_kernel_coefficients[8][second_coefficient_index];
  #pragma unroll
  for (int horner_term = 7; horner_term >= 0; horner_term--)
    second_coefficient_value = __fmaf_rn(
        horner_argument, second_coefficient_value,
        second_kernel_coefficients[horner_term][second_coefficient_index]);
  staged_point[coefficient_slot + 3] = second_coefficient_value;
}

// Support placement of a source in the frame of a group_size-cell lane group
// starting at lane_group_start, periodic in fine_grid_size_dimension. Returns
// the number of group cells the support covers (0, 1, ..., group_size), the
// kernel coefficient index of the first covered cell, and the local cell where
// coverage starts, so cell local_cell_offset + k uses coefficient
// support_coefficient_offset + k for k < group_cell_count.
//
// The support start can be a whole period or more below the lane group: a
// source folded near 0 has support_start = -SUPPORT_OVERHANG while a lane group
// near the high edge of the grid starts at grid_size - 2. The first add handles
// the common one-period case branchlessly and the loop handles the remaining
// periods for lane groups near the far edge. This matches the multi-period
// reduction in the library's owned-spread implementation.
__device__ __forceinline__ void place_support(int support_start,
                                              int fine_grid_size_dimension,
                                              int lane_group_start,
                                              int group_size,
                                              int &group_cell_count,
                                              int &support_coefficient_offset,
                                              int &local_cell_offset) {
  int relative_support_start = support_start - lane_group_start;
  relative_support_start += fine_grid_size_dimension & (relative_support_start >> 31);
  if (relative_support_start < 0) {
    relative_support_start += fine_grid_size_dimension;
    while (relative_support_start < 0)
      relative_support_start += fine_grid_size_dimension;
  }
  // group_cell_count / support_coefficient_offset / local_cell_offset below are
  // selects; ptxas lowers them to SEL/ISETP with no branch.
  const bool below_group = relative_support_start < group_size;
  const bool wraps_high =
      relative_support_start >= fine_grid_size_dimension - (WIDTH - 1);
  const int wrapped_cell_count =
      WIDTH - (fine_grid_size_dimension - relative_support_start);
  group_cell_count = below_group
                         ? (group_size - relative_support_start)
                         : (wraps_high ? (wrapped_cell_count < group_size
                                              ? wrapped_cell_count
                                              : group_size)
                                       : 0);
  support_coefficient_offset =
      below_group ? 0
                  : (wraps_high ? (fine_grid_size_dimension - relative_support_start) : 0);
  local_cell_offset = below_group ? relative_support_start : 0;
}

// Fast path for a support interval and a lane group that do not wrap around the
// periodic fine-grid boundary. The overlap is a plain interval intersection, so
// no modulo and no below_group/wraps_high predicates are needed.
__device__ __forceinline__ void place_support_nowrap(int support_start,
                                                    int lane_group_start,
                                                    int group_size,
                                                    int &group_cell_count,
                                                    int &support_coefficient_offset,
                                                    int &local_cell_offset) {
  const int intersection_begin = max(support_start, lane_group_start);
  const int intersection_end =
      min(support_start + WIDTH, lane_group_start + group_size);
  group_cell_count = max(intersection_end - intersection_begin, 0);
  support_coefficient_offset = intersection_begin - support_start;
  local_cell_offset = intersection_begin - lane_group_start;
}

// Periodic tap index of grid cell `cell` in the support of a source that starts
// at `support_start`, or -1 when the support does not reach the cell.
//
// fold_rescale returns [0, grid_size), so support_start = ceil(folded -
// ceil(WIDTH / 2)) is in [-ceil(WIDTH / 2), grid_size - ceil(WIDTH / 2)]. This
// is only called for cells in [0, grid_size), so cell - support_start is in
// (-grid_size, 2 * grid_size) and one conditional add plus one conditional
// subtract reduce it to [0, grid_size) with no loop and no division. Grid sizes
// are at least WIDTH, so at most one tap matches. The one-period bound needs a
// per-cell reference point, which is why the window path above (whose lane
// group can sit near the far edge of the grid) uses the multi-period loop in
// place_support instead.
__device__ __forceinline__ int periodic_tap_index(int cell, int support_start,
                                                  int grid_size) {
  int offset = cell - support_start;
  offset += (offset < 0) ? grid_size : 0;
  offset -= (offset >= grid_size) ? grid_size : 0;
  return offset < WIDTH ? offset : -1;
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
    const float (&z_kernel_values)[LANE_CELLS_Z], float strength_real,
    float strength_imag) {
  float xy_weights[LANE_CELLS_X][LANE_CELLS_Y];
  #pragma unroll
  for (int lx = 0; lx < LANE_CELLS_X; ++lx)
    #pragma unroll
    for (int ly = 0; ly < LANE_CELLS_Y; ++ly)
      xy_weights[lx][ly] = x_kernel_values[lx] * y_kernel_values[ly];

  // Scale z by the source strength once per lane group, so each of the 16 cell
  // updates is a single FMA per component instead of a strength * xy * z
  // multiply. exp_a keeps the z coefficients raw and scales here as well.
  float2 strength_scaled_z[LANE_CELLS_Z];
  #pragma unroll
  for (int lz = 0; lz < LANE_CELLS_Z; ++lz)
    strength_scaled_z[lz] = make_float2(strength_real * z_kernel_values[lz],
                                        strength_imag * z_kernel_values[lz]);

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

// Accumulate one staged source into this lane's cell group with exp_a's window
// reads: support placement, then six-plus-six-plus-six kernel coefficients
// loaded as three contiguous runs (six shared loads per dimension, each a warp
// broadcast). Kept out of line so the scan can issue two sources back to back
// and overlap their shared loads (ILP).
//
// The window reads hold the covered cells as one contiguous interval of the lane
// group. That needs grid_size > WIDTH + LANE_CELLS - 2: at or below that, the
// support can cover a group prefix and a group suffix at the same time (at
// grid_size == WIDTH it tiles the whole period), and one dynamic coefficient
// base cannot express the rotation. Such degenerate grids take
// scan_one_staged_point_periodic below.
template <bool SUPPORTS_DO_NOT_WRAP>
__device__ __forceinline__ void scan_one_staged_point(
    const float *__restrict__ staged_point, Grid grid, int lane_group_x_start,
    int lane_group_y_start, int lane_group_z_start, CellAccumulator &acc) {
  // Support starts were written as int bits by prepare_two_coefficients. The
  // record stride is 120 bytes, a multiple of 8, so this int2 load is aligned.
  const int2 support_start_xy = *reinterpret_cast<const int2 *>(staged_point);
  const int support_start_z = reinterpret_cast<const int *>(staged_point)[2];
  const float strength_real = staged_point[STRENGTH_SLOT];
  const float strength_imag = staged_point[STRENGTH_SLOT + 1];

  int group_cell_count_x, support_coefficient_offset_x, local_cell_offset_x;
  int group_cell_count_y, support_coefficient_offset_y, local_cell_offset_y;
  int group_cell_count_z, support_coefficient_offset_z, local_cell_offset_z;
  if constexpr (SUPPORTS_DO_NOT_WRAP) {
    place_support_nowrap(support_start_xy.x, lane_group_x_start, LANE_CELLS_X,
                         group_cell_count_x, support_coefficient_offset_x,
                         local_cell_offset_x);
    place_support_nowrap(support_start_xy.y, lane_group_y_start, LANE_CELLS_Y,
                         group_cell_count_y, support_coefficient_offset_y,
                         local_cell_offset_y);
    place_support_nowrap(support_start_z, lane_group_z_start, LANE_CELLS_Z,
                         group_cell_count_z, support_coefficient_offset_z,
                         local_cell_offset_z);
  } else {
    place_support(support_start_xy.x, grid.nx, lane_group_x_start, LANE_CELLS_X,
                  group_cell_count_x, support_coefficient_offset_x,
                  local_cell_offset_x);
    place_support(support_start_xy.y, grid.ny, lane_group_y_start, LANE_CELLS_Y,
                  group_cell_count_y, support_coefficient_offset_y,
                  local_cell_offset_y);
    place_support(support_start_z, grid.nz, lane_group_z_start, LANE_CELLS_Z,
                  group_cell_count_z, support_coefficient_offset_z,
                  local_cell_offset_z);
  }
  if (!(group_cell_count_x && group_cell_count_y && group_cell_count_z)) return;

  // x has a single pad (slot 11 serves as the y pre-pad), so its two values use
  // selects instead of a shifted base.
  const float x_kernel_value_0 =
      staged_point[X_COEFFICIENT_SLOT + support_coefficient_offset_x];
  const float x_kernel_value_1 =
      staged_point[X_COEFFICIENT_SLOT + 1 + support_coefficient_offset_x];
  const float x_kernel_values[LANE_CELLS_X] = {
      (local_cell_offset_x == 0) ? x_kernel_value_0 : 0.f,
      (local_cell_offset_x == 0) ? x_kernel_value_1 : x_kernel_value_0};
  // y needs no pre-pad beyond slot 11, so one shifted base gives both values.
  const int y_coefficient_base =
      Y_COEFFICIENT_SLOT + support_coefficient_offset_y - local_cell_offset_y;
  const float y_kernel_values[LANE_CELLS_Y] = {staged_point[y_coefficient_base],
                                              staged_point[y_coefficient_base + 1]};
  // The z row has three zero pre-pads and three zero post-pads, so the
  // coefficient for group cell local_z is simply the slot at
  // 21 + support_coefficient_offset_z - local_cell_offset_z + local_z and the
  // lane-group shift needs no per-value selects.
  const int z_coefficient_base =
      Z_COEFFICIENT_SLOT + support_coefficient_offset_z - local_cell_offset_z;
  const float z_kernel_values[LANE_CELLS_Z] = {
      staged_point[z_coefficient_base], staged_point[z_coefficient_base + 1],
      staged_point[z_coefficient_base + 2], staged_point[z_coefficient_base + 3]};

  accumulate_cell_group(acc, x_kernel_values, y_kernel_values, z_kernel_values,
                        strength_real, strength_imag);
}

// Degenerate-grid scan: per-cell periodic tap placement.
//
// Used when a grid dimension is too short for the window reads above, i.e.
// grid_size <= WIDTH + LANE_CELLS - 2 (x and y: 6, z: 8). Every cell then looks
// up its own tap index instead of reading a window, so a wrapped support costs
// selects rather than a different layout. Those grids are at most a few cells
// wide (the bench's 6x6x6 and 6x9x13 cases), so this path is never hot; it is
// also correct for the wide dimensions of the same launch, which is why one flag
// selects it for the whole scan.
__device__ __forceinline__ void scan_one_staged_point_periodic(
    const float *__restrict__ staged_point, Grid grid, int lane_group_x_start,
    int lane_group_y_start, int lane_group_z_start, CellAccumulator &acc) {
  const int2 support_start_xy = *reinterpret_cast<const int2 *>(staged_point);
  const int support_start_z = reinterpret_cast<const int *>(staged_point)[2];
  const float strength_real = staged_point[STRENGTH_SLOT];
  const float strength_imag = staged_point[STRENGTH_SLOT + 1];

  // Per-cell taps: a cell the support does not reach keeps a zero weight.
  float x_kernel_values[LANE_CELLS_X];
  float y_kernel_values[LANE_CELLS_Y];
  float z_kernel_values[LANE_CELLS_Z];
  int covered_x = 0, covered_y = 0, covered_z = 0;
  #pragma unroll
  for (int lx = 0; lx < LANE_CELLS_X; ++lx) {
    const int tap = periodic_tap_index(lane_group_x_start + lx, support_start_xy.x,
                                       grid.nx);
    x_kernel_values[lx] =
        (tap >= 0) ? staged_point[X_COEFFICIENT_SLOT + tap] : 0.f;
    covered_x += (tap >= 0);
  }
  #pragma unroll
  for (int ly = 0; ly < LANE_CELLS_Y; ++ly) {
    const int tap = periodic_tap_index(lane_group_y_start + ly, support_start_xy.y,
                                       grid.ny);
    y_kernel_values[ly] =
        (tap >= 0) ? staged_point[Y_COEFFICIENT_SLOT + tap] : 0.f;
    covered_y += (tap >= 0);
  }
  #pragma unroll
  for (int lz = 0; lz < LANE_CELLS_Z; ++lz) {
    const int tap =
        periodic_tap_index(lane_group_z_start + lz, support_start_z, grid.nz);
    z_kernel_values[lz] =
        (tap >= 0) ? staged_point[Z_COEFFICIENT_SLOT + tap] : 0.f;
    covered_z += (tap >= 0);
  }
  // A dimension the support misses entirely zeroes every cell.
  if (!(covered_x && covered_y && covered_z)) return;

  accumulate_cell_group(acc, x_kernel_values, y_kernel_values, z_kernel_values,
                        strength_real, strength_imag);
}

// Accumulate every staged source of the current batch. Two sources per
// iteration let the compiler overlap their shared loads and FMAs.
template <bool SUPPORTS_DO_NOT_WRAP, bool PERIODIC_TAPS>
__device__ __forceinline__ void scan_staged_batch(
    const float *staging, int staged_point_count, Grid grid,
    int lane_group_x_start, int lane_group_y_start, int lane_group_z_start,
    CellAccumulator &acc) {
  int staged_point_slot = 0;
  for (; staged_point_slot + 1 < staged_point_count; staged_point_slot += 2) {
    #pragma unroll
    for (int pair = 0; pair < 2; ++pair) {
      const float *staged_point =
          staging + (staged_point_slot + pair) * VALUES_PER_POINT;
      if constexpr (PERIODIC_TAPS)
        scan_one_staged_point_periodic(staged_point, grid, lane_group_x_start,
                                       lane_group_y_start, lane_group_z_start,
                                       acc);
      else
        scan_one_staged_point<SUPPORTS_DO_NOT_WRAP>(
            staged_point, grid, lane_group_x_start, lane_group_y_start,
            lane_group_z_start, acc);
    }
  }
  if (staged_point_slot < staged_point_count) {
    const float *staged_point = staging + staged_point_slot * VALUES_PER_POINT;
    if constexpr (PERIODIC_TAPS)
      scan_one_staged_point_periodic(staged_point, grid, lane_group_x_start,
                                     lane_group_y_start, lane_group_z_start, acc);
    else
      scan_one_staged_point<SUPPORTS_DO_NOT_WRAP>(
          staged_point, grid, lane_group_x_start, lane_group_y_start,
          lane_group_z_start, acc);
  }
}

// Write this lane's cell group, skipping cells past the clipped warp-region
// extent. Cells outside the grid are scanned with garbage support placement and
// are never stored.
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

// Active detection: one thread per warp region. A region is active when the sum
// of the source counts of its candidate bins is nonzero. The candidate-bin set is
// exactly the one SourceRangeCursor enumerates, so the test can only
// over-report: a source in a candidate bin may still miss the region, but no
// region that would accumulate a contribution is ever skipped.
__global__ void detect_active_regions(const int *__restrict__ bin_offsets,
                                      Grid grid, int regions_x, int regions_y,
                                      int region_count,
                                      int *__restrict__ active_flags) {
  const int region_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (region_id >= region_count) return;
  const WarpRegion region =
      warp_region_from_id(region_id, grid, regions_x, regions_y);
  if (region.nx <= 0 || region.ny <= 0 || region.nz <= 0) {
    active_flags[region_id] = 0;
    return;
  }
  const detail::CandidateBins x_bins = detail::candidate_bins(
      region.x, region.x + region.nx, grid.bin_size, grid.nx, grid.bins_x);
  const detail::CandidateBins y_bins = detail::candidate_bins(
      region.y, region.y + region.ny, grid.bin_size, grid.ny, grid.bins_y);
  const detail::CandidateBins z_bins = detail::candidate_bins(
      region.z, region.z + region.nz, grid.bin_size, grid.nz, grid.bins_z);
  int source_count = 0;
  for (int z_index = 0; z_index < z_bins.count; ++z_index)
    for (int y_index = 0; y_index < y_bins.count; ++y_index)
      for (int x_index = 0; x_index < x_bins.count; ++x_index) {
        const int bin =
            x_bins[x_index] +
            grid.bins_x * (y_bins[y_index] + grid.bins_y * z_bins[z_index]);
        source_count += bin_offsets[bin + 1] - bin_offsets[bin];
      }
  active_flags[region_id] = source_count != 0 ? 1 : 0;
}

// Compact the active region IDs in region order. Single-threaded on purpose: the
// region count is small (512 for 64^3, 4096 for 128^3) and one thread keeps the
// list deterministic without atomics.
__global__ void compact_active_regions(const int *__restrict__ active_flags,
                                       int region_count,
                                       int *__restrict__ active_regions,
                                       int *__restrict__ active_count) {
  int active_total = 0;
  for (int region_id = 0; region_id < region_count; ++region_id)
    if (active_flags[region_id]) active_regions[active_total++] = region_id;
  active_count[0] = active_total;
}

// Sum the per-split partial grids into the output for active warp regions only.
// Inactive regions are zeroed once by the host memset and are never read here, so
// a split launch reduces split_count * active_count * WARP_REGION_CELLS cells
// instead of split_count whole grids. Each active-region cell of each split slice
// is fully defined by the spread kernel, which always stores its whole region.
__global__ void sum_splits(const cuFloatComplex *__restrict__ partial,
                           int split_count, long long cells,
                           cuFloatComplex *__restrict__ output,
                           const int *__restrict__ active_regions,
                           int active_count, Grid grid, int regions_x,
                           int regions_y) {
  const long long task_count = (long long)active_count * WARP_REGION_CELLS;
  const float2 *const split_values = reinterpret_cast<const float2 *>(partial);
  for (long long task = (long long)blockIdx.x * blockDim.x + threadIdx.x;
       task < task_count; task += (long long)blockDim.x * gridDim.x) {
    const int region_slot = (int)(task / WARP_REGION_CELLS);
    const int local_cell = (int)(task - (long long)region_slot * WARP_REGION_CELLS);
    const int local_x = local_cell % WARP_REGION_X;
    const int local_y = (local_cell / WARP_REGION_X) % WARP_REGION_Y;
    const int local_z = local_cell / (WARP_REGION_X * WARP_REGION_Y);
    const WarpRegion region = warp_region_from_id(active_regions[region_slot], grid,
                                                  regions_x, regions_y);
    if (local_x >= region.nx || local_y >= region.ny || local_z >= region.nz)
      continue;
    const int x = region.x + local_x;
    const int y = region.y + local_y;
    const int z = region.z + local_z;
    const long long index = x + (long long)grid.nx * (y + (long long)grid.ny * z);
    float2 value = split_values[index];
    for (int split = 1; split < split_count; ++split) {
      const float2 split_value = split_values[(long long)split * cells + index];
      value.x += split_value.x;
      value.y += split_value.y;
    }
    output[index] = make_cuFloatComplex(value.x, value.y);
  }
}

// Coefficient-staging geometry for one lane, loop-invariant over a whole launch:
// lanes 0..26 split into three groups of nine, one group per staged source slot,
// and within a group lane % 3 picks the kernel coefficient index while lane / 3
// picks the dimension. Lanes 27..31 are idle during coefficient preparation but
// still participate in every __syncwarp().
struct StagingLane {
  int grid_size_dimension;
  int support_start_slot;
  int coefficient_slot;
  int kernel_coefficient_index;
};

__device__ __forceinline__ StagingLane make_staging_lane(int lane, Grid grid) {
  const int coefficient_lane = lane % 9;
  const int dimension = coefficient_lane / 3;
  const int kernel_coefficient_index = coefficient_lane % 3;
  const int grid_size_dimension = (dimension == 0)
                                      ? grid.nx
                                      : ((dimension == 1) ? grid.ny : grid.nz);
  const int support_start_slot = SUPPORT_START_SLOT + dimension;
  const int coefficient_slot =
      (dimension == 0)
          ? (X_COEFFICIENT_SLOT + kernel_coefficient_index)
          : (dimension == 1)
                ? (Y_COEFFICIENT_SLOT + kernel_coefficient_index)
                : (Z_COEFFICIENT_SLOT + kernel_coefficient_index);
  return StagingLane{grid_size_dimension, support_start_slot, coefficient_slot,
                     kernel_coefficient_index};
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
    int lane, float *__restrict__ warp_staging,
    const float (&kernel_coefficients)[9],
    const float (*second_kernel_coefficients)[WIDTH],
    const StagingLane &staging_lane) {
  const int lane_group_x_start = region.x + cell_group.x;
  const int lane_group_y_start = region.y + cell_group.y;
  const int lane_group_z_start = region.z + cell_group.z;
  // Only a fully interior warp region can use the branch-free support path. A
  // clipped region keeps lane groups past the grid edge, whose support
  // placement is meaningless but never stored.
  const bool warp_region_inside_grid = region.nx == WARP_REGION_X &&
                                       region.ny == WARP_REGION_Y &&
                                       region.nz == WARP_REGION_Z;
  // A dimension whose period is at most WIDTH + LANE_CELLS - 2 makes the
  // contiguous coefficient window ambiguous (see scan_one_staged_point_periodic).
  // This is a whole-launch property, so it is loop-invariant and warp-uniform.
  const bool narrow_grid = grid.nx <= WIDTH + LANE_CELLS_X - 2 ||
                           grid.ny <= WIDTH + LANE_CELLS_Y - 2 ||
                           grid.nz <= WIDTH + LANE_CELLS_Z - 2;

  CellAccumulator acc = {};

  // Zero the padded coefficient slots once per warp. Neither the raw pass nor the
  // coefficient pass ever writes these slots again, so scan_one_staged_point can
  // rely on them staying zero for the whole kernel: the x/y/z dynamic coefficient
  // bases depend on that (a base may point at a pad when the lane group covers
  // fewer than LANE_CELLS_* cells of the support).
  for (int point_slot = lane; point_slot < MAX_POINTS_PER_BATCH; point_slot += 32) {
    float *const staged_point = warp_staging + point_slot * VALUES_PER_POINT;
    staged_point[X_Y_PAD_SLOT] = 0.f;
    #pragma unroll
    for (int pad = 0; pad < Z_PADDING_BEFORE; ++pad)
      staged_point[Z_COEFFICIENT_SLOT - Z_PADDING_BEFORE + pad] = 0.f;
    #pragma unroll
    for (int pad = 0; pad < Z_PADDING_AFTER; ++pad)
      staged_point[Z_COEFFICIENT_SLOT + WIDTH + pad] = 0.f;
  }
  __syncwarp();

  const int point_batch_stride = split_count * MAX_POINTS_PER_BATCH;

  SourceRangeCursor cursor(region, grid, bin_offsets);
  int begin, end;
  while (cursor.next(begin, end)) {
    for (int first_point_index = begin + split_index * MAX_POINTS_PER_BATCH;
         first_point_index < end; first_point_index += point_batch_stride) {
      const int staged_point_count =
          min(MAX_POINTS_PER_BATCH, end - first_point_index);
      stage_raw_points(points, first_point_index, staged_point_count, lane,
                       warp_staging);
      __syncwarp();

      // Coefficient pass: nine lanes per source rewrite the three support starts
      // in place as ints and evaluate the source's 18 kernel coefficients.
      if (lane < 27) {
        const int point_slot = lane / 9;
        float *staged_point = warp_staging + point_slot * VALUES_PER_POINT;
        for (int slot = point_slot; slot < staged_point_count;
             slot += 3, staged_point += 3 * VALUES_PER_POINT)
          prepare_two_coefficients(
              staged_point, staging_lane.grid_size_dimension,
              staging_lane.support_start_slot, staging_lane.coefficient_slot,
              staging_lane.kernel_coefficient_index + 3, kernel_coefficients,
              second_kernel_coefficients);
      }
      __syncwarp();

      // One warp-wide decision per batch: when the warp region is interior and
      // every staged source has a non-wrapping support, the whole batch uses the
      // branch-free min/max support path; otherwise it uses the general periodic
      // path. The vote runs for all lanes, including lanes past count.
      bool support_inside_grid = true;
      if (lane < staged_point_count) {
        const int *const support_start_bits =
            reinterpret_cast<const int *>(warp_staging + lane * VALUES_PER_POINT);
        const int support_start_x = support_start_bits[SUPPORT_START_SLOT];
        const int support_start_y = support_start_bits[SUPPORT_START_SLOT + 1];
        const int support_start_z = support_start_bits[SUPPORT_START_SLOT + 2];
        support_inside_grid =
            support_start_x >= 0 && support_start_y >= 0 && support_start_z >= 0 &&
            support_start_x + WIDTH <= grid.nx &&
            support_start_y + WIDTH <= grid.ny &&
            support_start_z + WIDTH <= grid.nz;
      }
      const bool whole_batch_nonwrapping =
          warp_region_inside_grid && __all_sync(0xffffffff, support_inside_grid);

      if (narrow_grid) {
        // Degenerate narrow dimension: per-cell periodic tap placement. Checked
        // first because a narrow z dimension can still be fully interior.
        scan_staged_batch<false, true>(warp_staging, staged_point_count, grid,
                                       lane_group_x_start, lane_group_y_start,
                                       lane_group_z_start, acc);
      } else if (whole_batch_nonwrapping) {
        scan_staged_batch<true, false>(warp_staging, staged_point_count, grid,
                                       lane_group_x_start, lane_group_y_start,
                                       lane_group_z_start, acc);
      } else {
        scan_staged_batch<false, false>(warp_staging, staged_point_count, grid,
                                        lane_group_x_start, lane_group_y_start,
                                        lane_group_z_start, acc);
      }
      // Staging invariant: every lane finishes reading the single shared buffer
      // before the next batch overwrites slots 0..26 of its records.
      __syncwarp();
    }
  }

  store_cell_group(acc, region_output, grid, region, cell_group);
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
            int work_count) {
  // One single-buffered staging group per warp: MAX_POINTS_PER_BATCH 30-float
  // staged-source records plus the shared coefficient copy. The buffer is
  // 16-byte aligned so the 120-byte record stride keeps the int2 support-start
  // load aligned.
  __shared__ __align__(16) float staged_point_values[WARPS_PER_THREADBLOCK]
                                                     [MAX_POINTS_PER_BATCH]
                                                     [VALUES_PER_POINT];
  __shared__ float second_kernel_coefficients[9][WIDTH];

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const StagingLane staging_lane = make_staging_lane(lane, grid);

  float kernel_coefficients[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if (lane < 27) {
    #pragma unroll
    for (int kernel_term = 0; kernel_term < 9; kernel_term++)
      kernel_coefficients[kernel_term] =
          KERNEL_COEFFICIENT_TABLE[kernel_term]
                                  [staging_lane.kernel_coefficient_index];
  }
  for (int table_entry = threadIdx.x; table_entry < 9 * WIDTH;
       table_entry += blockDim.x)
    second_kernel_coefficients[table_entry / WIDTH][table_entry % WIDTH] =
        KERNEL_COEFFICIENT_TABLE[table_entry / WIDTH][table_entry % WIDTH];
  // The only threadblock-wide barrier in the kernel: it makes the shared
  // coefficient copy complete for every warp. It runs before the work-item
  // bounds check below, so no warp can miss it. Everything after it is
  // warp-local, so the early returns cannot strand a threadblock barrier.
  __syncthreads();

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

  // A split writes its whole warp region into its own slice of the partial grid;
  // the single-split paths write the final grid directly.
  cuFloatComplex *const region_output =
      (split_count > 1)
          ? output + (long long)split_index * cell_count(grid)
          : output;

  spread_region(points, bin_offsets, grid, region, lane_cell_group(lane),
                region_output, split_index, split_count, lane,
                &staged_point_values[warp][0][0], kernel_coefficients,
                second_kernel_coefficients, staging_lane);
}

}  // namespace

void launch_assign_bins(const Point *points, int count, Grid grid, int *bin_ids) {
  constexpr int THREADS = 256;
  assign_bins<<<(count + THREADS - 1) / THREADS, THREADS>>>(points, count, grid, bin_ids);
}

void launch_spread(const Point *points, const int *bin_offsets, Grid grid,
                   cuFloatComplex *output) {
  const int regions_x = warp_regions_x(grid);
  const int regions_y = warp_regions_y(grid);
  const int region_count = warp_region_count(grid);
  const long long cells = cell_count(grid);
  const size_t output_bytes = (size_t)cells * sizeof(cuFloatComplex);

  // Temporaries are allocated per launch with cudaMalloc/cudaFree; there is no
  // persistent workspace yet. The benchmark's spread interval therefore includes
  // the two metadata kernels, the blocking host copy of active_count, the output
  // memset, the partial-grid allocation, and the split reduction. Those costs are
  // reported rather than hidden.
  int *active_flags = nullptr;
  int *active_regions = nullptr;
  int *active_count = nullptr;
  check_cuda(cudaMalloc(&active_flags, (size_t)region_count * sizeof(int)),
             "cudaMalloc(active_flags)");
  check_cuda(cudaMalloc(&active_regions, (size_t)region_count * sizeof(int)),
             "cudaMalloc(active_regions)");
  check_cuda(cudaMalloc(&active_count, sizeof(int)), "cudaMalloc(active_count)");

  constexpr int METADATA_THREADS = 256;
  detect_active_regions<<<(region_count + METADATA_THREADS - 1) / METADATA_THREADS,
                          METADATA_THREADS>>>(bin_offsets, grid, regions_x,
                                              regions_y, region_count, active_flags);
  compact_active_regions<<<1, 1>>>(active_flags, region_count, active_regions,
                                   active_count);
  int host_active_count = 0;
  check_cuda(cudaMemcpy(&host_active_count, active_count, sizeof(int),
                        cudaMemcpyDeviceToHost),
             "cudaMemcpy(active_count)");

  // exp_a's split heuristic: aim for about 2048 work items when few regions are
  // active (clustered and dense inputs), otherwise 512, and never exceed 32.
  // Uniform and sparse inputs stay at split_count == 1.
  const int split_target = host_active_count < 512 ? 2048 : 512;
  int split_count = 1;
  while (split_count < 32 && host_active_count * split_count < split_target)
    split_count <<= 1;

  const dim3 threads(THREADS_PER_THREADBLOCK);
  cuFloatComplex *partial = nullptr;

  if (host_active_count == 0) {
    // No warp region has a source: every cell is a zero.
    check_cuda(cudaMemsetAsync(output, 0, output_bytes), "cudaMemsetAsync(output)");
  } else if (split_count > 1) {
    // Split path: each work item writes its whole warp region into its own split
    // slice, so every active-region cell of every slice is defined.
    const int work_count = host_active_count * split_count;
    check_cuda(cudaMalloc(&partial, (size_t)split_count * output_bytes),
               "cudaMalloc(partial)");
    spread<<<(work_count + WARPS_PER_THREADBLOCK - 1) / WARPS_PER_THREADBLOCK,
             threads>>>(points, bin_offsets, grid, partial, active_regions,
                        split_count, work_count);
    // Zero the inactive regions once; the reduction writes only active cells.
    check_cuda(cudaMemsetAsync(output, 0, output_bytes), "cudaMemsetAsync(output)");
    const long long reduce_tasks = (long long)host_active_count * WARP_REGION_CELLS;
    const long long reduce_blocks =
        (reduce_tasks + METADATA_THREADS - 1) / METADATA_THREADS;
    sum_splits<<<(unsigned int)reduce_blocks, METADATA_THREADS>>>(
        partial, split_count, cells, output, active_regions, host_active_count,
        grid, regions_x, regions_y);
    check_cuda(cudaFree(partial), "cudaFree(partial)");
  } else if (host_active_count == region_count) {
    // Full-grid fast path: every region is active, so the active list is only an
    // identity map. Skip it and write the output directly with no memset.
    spread<<<(region_count + WARPS_PER_THREADBLOCK - 1) / WARPS_PER_THREADBLOCK,
             threads>>>(points, bin_offsets, grid, output, nullptr, 1,
                        region_count);
  } else {
    // Active-list single-split path: skip empty regions, but zero the output first
    // because those regions are never written.
    check_cuda(cudaMemsetAsync(output, 0, output_bytes), "cudaMemsetAsync(output)");
    spread<<<(host_active_count + WARPS_PER_THREADBLOCK - 1) /
                 WARPS_PER_THREADBLOCK,
             threads>>>(points, bin_offsets, grid, output, active_regions, 1,
                        host_active_count);
  }

  check_cuda(cudaFree(active_flags), "cudaFree(active_flags)");
  check_cuda(cudaFree(active_regions), "cudaFree(active_regions)");
  check_cuda(cudaFree(active_count), "cudaFree(active_count)");
}

}  // namespace atomic_free_spread
