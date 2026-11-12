// Standalone atomic-free 3D type-1 spread: FP32, width 6, sigma 2, periodic.
// Each thread owns output cells and writes them once, without spreading atomics.
// GPU bin assignment uses the same fold_rescale as spreading. The benchmark
// sorts with GPU stable radix by default, with a CPU counting-sort fallback.
// Timing and validation live in atomic_free_spread_bench.cu.
// Build/run: ./devel/morton_spread/run_atomic_free_spread.sh.

#include "atomic_free_spread.cuh"

#include <cufinufft/spreadinterp.hpp>

#include <cmath>
#include <type_traits>

namespace atomic_free_spread {
namespace {

// Align float rows for vector loads; padding spreads cells across bank groups.
constexpr int WEIGHT_STRIDE = STAGE_POINTS + 4;

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

// Fill tile-local weights, zero outside support. The float2 z rows include strength.
template<int COUNT, typename T>
__device__ __forceinline__ void stage_tile_weights(T (*dest)[WEIGHT_STRIDE],
                                                   int point, int tile_begin,
                                                   int start, int fine_size,
                                                   const float *ker, float2 strength = {}) {
  #pragma unroll
  for (int local = 0; local < COUNT; ++local)
    dest[local][point] = T{};

  // start is in [-HALO, fine_size - HALO]; one add normalizes it.
  start += (start < 0) ? fine_size : 0;

  #pragma unroll
  for (int k = 0; k < WIDTH; ++k) {
    int cell = start + k;
    cell -= (cell >= fine_size) ? fine_size : 0;
    const int local = cell - tile_begin;
    if (local >= 0 && local < COUNT) {
      if constexpr (std::is_same_v<T, float2>)
        dest[local][point] = make_float2(ker[k] * strength.x, ker[k] * strength.y);
      else
        dest[local][point] = ker[k];
    }
  }
}

// Add the staged batch to this thread's cell group. Each x-y product serves the
// group's z cells and each z value serves its x cells, so a point costs
// GROUP_X + GROUP_Y + GROUP_Z shared loads for GROUP_X * GROUP_Y * GROUP_Z cells.
__device__ __forceinline__ void accumulate_batch(
    const float (*kx)[WEIGHT_STRIDE], const float (*ky)[WEIGHT_STRIDE],
    const float2 (*kz)[WEIGHT_STRIDE], int staged, int x0, int y0, int z0,
    float2 (&acc)[GROUP_X][GROUP_Y][GROUP_Z]) {
  for (int j = 0; j < staged; ++j) {
    float weight[GROUP_X][GROUP_Y];
    #pragma unroll
    for (int gx = 0; gx < GROUP_X; ++gx)
      #pragma unroll
      for (int gy = 0; gy < GROUP_Y; ++gy)
        weight[gx][gy] = kx[x0 + gx][j] * ky[y0 + gy][j];

    float2 z[GROUP_Z];
    #pragma unroll
    for (int gz = 0; gz < GROUP_Z; ++gz) z[gz] = kz[z0 + gz][j];

    #pragma unroll
    for (int gx = 0; gx < GROUP_X; ++gx)
      #pragma unroll
      for (int gy = 0; gy < GROUP_Y; ++gy)
        #pragma unroll
        for (int gz = 0; gz < GROUP_Z; ++gz) {
          acc[gx][gy][gz].x += z[gz].x * weight[gx][gy];
          acc[gx][gy][gz].y += z[gz].y * weight[gx][gy];
        }
  }
}

// Write this thread's cell group, skipping cells past the end of a partial tile.
__device__ __forceinline__ void store_cell_group(
    cuFloatComplex *output, Grid grid, int tile_x, int tile_y, int tile_z, int x0,
    int y0, int z0, const float2 (&acc)[GROUP_X][GROUP_Y][GROUP_Z]) {
  #pragma unroll
  for (int gx = 0; gx < GROUP_X; ++gx)
    #pragma unroll
    for (int gy = 0; gy < GROUP_Y; ++gy)
      #pragma unroll
      for (int gz = 0; gz < GROUP_Z; ++gz) {
        const int x = tile_x + x0 + gx;
        const int y = tile_y + y0 + gy;
        const int z = tile_z + z0 + gz;
        if (x < grid.nx && y < grid.ny && z < grid.nz) {
          const long long index =
              x + (long long)grid.nx * (y + (long long)grid.ny * z);
          output[index] = make_cuFloatComplex(acc[gx][gy][gz].x, acc[gx][gy][gz].y);
        }
      }
}

// One block owns a tile and stages zero-filled weights for its thread cell groups.
__global__ void spread(const Point *points, const int *bin_offsets, Grid grid,
                       cuFloatComplex *output) {
  using cufinufft::spreadinterp::eval_kernel_vec_horner;
  using cufinufft::spreadinterp::fold_rescale;

  int tile_x = blockIdx.x * TILE_X;
  int tile_y = blockIdx.y * TILE_Y;
  int tile_z = blockIdx.z * TILE_Z;
  int x0 = threadIdx.x * GROUP_X;
  int y0 = threadIdx.y * GROUP_Y;
  int z0 = threadIdx.z * GROUP_Z;

  __shared__ detail::NearBins x_bins, y_bins, z_bins;
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    const int end_x = min(tile_x + TILE_X, grid.nx);
    const int end_y = min(tile_y + TILE_Y, grid.ny);
    const int end_z = min(tile_z + TILE_Z, grid.nz);
    x_bins = detail::near_bins(tile_x, end_x, grid.bin_size, grid.nx, grid.bins_x);
    y_bins = detail::near_bins(tile_y, end_y, grid.bin_size, grid.ny, grid.bins_y);
    z_bins = detail::near_bins(tile_z, end_z, grid.bin_size, grid.nz, grid.bins_z);
  }

  // Keep consecutive points contiguous so the scan can use vector loads.
  // kx[local][j] is point j's weight for tile_x + local (likewise y); kz also
  // includes the strength. Padding separates the warp's two z slices into
  // different bank groups.
  __shared__ __align__(16) float kx[TILE_X][WEIGHT_STRIDE];
  __shared__ __align__(16) float ky[TILE_Y][WEIGHT_STRIDE];
  __shared__ __align__(16) float2 kz[TILE_Z][WEIGHT_STRIDE];
  __syncthreads();

  const int linear_tid =
      threadIdx.x + BLOCK_X * (threadIdx.y + BLOCK_Y * threadIdx.z);

  // One staging lane per (point, dimension): dim d owns the lanes
  // [d * STAGE_POINTS, (d + 1) * STAGE_POINTS), so every warp stages a single
  // dimension and the dimension branches below do not diverge. Lanes past
  // 3 * STAGE_POINTS only take part in the barriers.
  const int stage_dim = linear_tid / STAGE_POINTS;
  const int stage_point = linear_tid % STAGE_POINTS;
  const bool is_stager = stage_dim < 3;

  float2 acc[GROUP_X][GROUP_Y][GROUP_Z] = {};

  for (int z_bin_index = 0; z_bin_index < z_bins.count; ++z_bin_index) {
    const int bz = z_bins[z_bin_index];
    // Skip accumulation when a bin's periodic support misses this half-warp's z slice.
    int dz = tile_z + z0 - bz * grid.bin_size;
    if (dz + GROUP_Z <= -HALO) dz += grid.nz;
    if (dz + GROUP_Z > grid.nz - HALO) dz -= grid.nz;
    const bool covered_z = dz < min(grid.bin_size, grid.nz - bz * grid.bin_size) + HALO;
    for (int y_bin_index = 0; y_bin_index < y_bins.count; ++y_bin_index) {
      const int by = y_bins[y_bin_index];
      for (int x_bin_index = 0; x_bin_index < x_bins.count;) {
        const int bx = x_bins[x_bin_index];
        // Adjacent x bins form one contiguous point range; stop at the seam.
        const int run = min(x_bins.count - x_bin_index, grid.bins_x - bx);
        x_bin_index += run;
        const int bin = bx + grid.bins_x * (by + grid.bins_y * bz);
        const int bin_begin = bin_offsets[bin];
        const int bin_end = bin_offsets[bin + run];
        // This loop carries the block's barriers, so every thread must run it;
        // only staging and accumulation are conditional.
        for (int base = bin_begin; base < bin_end; base += STAGE_POINTS) {
          const int staged = min(STAGE_POINTS, bin_end - base);
          if (is_stager && stage_point < staged) {
            const Point p = points[base + stage_point];
            const float raw = stage_dim == 0 ? p.x : (stage_dim == 1 ? p.y : p.z);
            const int size =
                stage_dim == 0 ? grid.nx : (stage_dim == 1 ? grid.ny : grid.nz);
            const float rescaled = fold_rescale(raw, size);
            const int start = int(ceilf(rescaled - HALO));
            float ker[WIDTH];
            eval_kernel_vec_horner<float, WIDTH>(ker, float(start) - rescaled, 2.0);
            if (stage_dim == 0) {
              stage_tile_weights<TILE_X>(kx, stage_point, tile_x, start, grid.nx, ker);
            } else if (stage_dim == 1) {
              stage_tile_weights<TILE_Y>(ky, stage_point, tile_y, start, grid.ny, ker);
            } else {
              stage_tile_weights<TILE_Z>(kz, stage_point, tile_z, start, grid.nz, ker,
                                         make_float2(p.strength.x, p.strength.y));
            }
          }
          __syncthreads();
          if (covered_z) {
            // Unroll full batches at compile time; keep partial batches compact.
            if (staged == STAGE_POINTS)
              accumulate_batch(kx, ky, kz, STAGE_POINTS, x0, y0, z0, acc);
            else
              accumulate_batch(kx, ky, kz, staged, x0, y0, z0, acc);
          }
          __syncthreads();
        }
      }
    }
  }
  store_cell_group(output, grid, tile_x, tile_y, tile_z, x0, y0, z0, acc);
}

}  // namespace

void launch_assign_bins(const Point *points, int count, Grid grid, int *bin_ids) {
  constexpr int THREADS = 256;
  assign_bins<<<(count + THREADS - 1) / THREADS, THREADS>>>(points, count, grid, bin_ids);
}

void launch_spread(const Point *points, const int *bin_offsets, Grid grid,
                   cuFloatComplex *output) {
  dim3 threads(BLOCK_X, BLOCK_Y, BLOCK_Z);
  dim3 blocks(1 + (grid.nx - 1) / TILE_X, 1 + (grid.ny - 1) / TILE_Y,
              1 + (grid.nz - 1) / TILE_Z);
  spread<<<blocks, threads>>>(points, bin_offsets, grid, output);
}

}  // namespace atomic_free_spread
