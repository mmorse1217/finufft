// F0 point-owned 3D type-1 spreading prototype: implementation.
//
// This is a standalone experiment, not part of the library build. One CUDA
// block owns one exact output tile. One thread processes one point at a time:
// it folds the point, clips the point's width-6 support to the tile, evaluates
// the three Horner weight vectors into a per-thread column of shared scratch,
// and scatters the support into two shared tile arrays with shared-memory
// atomics. After all bins have been visited, the block writes every in-grid
// tile cell exactly once, without atomics.
//
// There are no global atomics and no barriers inside the point loop. Because
// many lanes add into the same shared cells, the summation order within a tile
// is nondeterministic; callers must compare with a tolerance rather than
// expecting bitwise reproducibility. The caller must check cudaGetLastError()
// after launch_spread_f0().
//
// Build and run with ./devel/morton_spread/run_atomic_free_spread.sh KERNEL=1.

#include "atomic_free_spread.cuh"

#include <cufinufft/spreadinterp.hpp>

#include <cmath>

namespace atomic_free_spread {
namespace f0 {

// One block owns an exact TILE_X x TILE_Y x TILE_Z output tile. PAD_X makes the
// x stride odd, so the shared tile's bank mapping rotates across rows instead
// of aligning every row on the same banks.
constexpr int TILE_X = 8;
constexpr int TILE_Y = 8;
constexpr int TILE_Z = 4;
constexpr int THREADS = 128;
constexpr int PAD_X = TILE_X + 1;
constexpr int TILE_CELLS = TILE_X * TILE_Y * TILE_Z;
constexpr int TILE_FLOATS = PAD_X * TILE_Y * TILE_Z;

static_assert(THREADS % 32 == 0, "a block must contain whole warps");
static_assert(TILE_CELLS == 2 * THREADS, "each thread writes two tile cells");
static_assert(PAD_X >= TILE_X + 1, "the x stride must be padded");

// A folded point plus the integer starts of its width-WIDTH support on each
// axis, matching the baseline staging expression ceilf(folded - SUPPORT_OVERHANG).
struct FoldedPoint {
  float x, y, z;
  int sx, sy, sz;
};

struct TileOrigin {
  int x, y, z;
};

// One contiguous run of support cells inside this tile. begin/end are
// tile-local coordinates; weight_index is the Horner weight index of begin.
struct TileSpan {
  int begin, end, weight_index;
};

struct SupportSpans {
  TileSpan x[2], y[2], z[2];
  int count_x, count_y, count_z;
};

// Fold and rescale each coordinate once, then derive the support start exactly
// as the baseline does.
__device__ __forceinline__ FoldedPoint fold_point(const Point &point, Grid grid) {
  using cufinufft::spreadinterp::fold_rescale;
  FoldedPoint folded;
  folded.x = fold_rescale(point.x, grid.nx);
  folded.y = fold_rescale(point.y, grid.ny);
  folded.z = fold_rescale(point.z, grid.nz);
  folded.sx = int(ceilf(folded.x - SUPPORT_OVERHANG));
  folded.sy = int(ceilf(folded.y - SUPPORT_OVERHANG));
  folded.sz = int(ceilf(folded.z - SUPPORT_OVERHANG));
  return folded;
}

// Clip one dimension's support [start, start + WIDTH) to [tile_begin,
// tile_end) in [0, grid_size), splitting at the periodic seam. The support can
// cross at most one period boundary because grid_size >= WIDTH, so at most two
// spans are emitted. This is the only place that does wrap/split arithmetic.
//
// For each period shift, intersect the unwrapped support with
// [shift, shift + grid_size), wrap back to [0, grid_size), clip to the tile,
// and record the weight index of the first surviving cell. The weight index is
// always in [0, WIDTH), and the span length never exceeds WIDTH.
__host__ __device__ __forceinline__ int clip_support_to_tile(
    int start, int grid_size, int tile_begin, int tile_end, TileSpan spans[2]) {
  const int support_end = start + WIDTH;
  int count = 0;
  #pragma unroll
  for (int period = -1; period <= 1; ++period) {
    const int shift = period * grid_size;
    const int low = shift > start ? shift : start;
    const int high = shift + grid_size < support_end ? shift + grid_size
                                                     : support_end;
    if (low >= high) continue;

    // Wrap the unwrapped intersection [low, high) into [0, grid_size).
    const int wrapped_begin = low - shift;
    const int wrapped_end = high - shift;
    const int clipped_begin =
        wrapped_begin > tile_begin ? wrapped_begin : tile_begin;
    const int clipped_end = wrapped_end < tile_end ? wrapped_end : tile_end;
    if (clipped_begin >= clipped_end) continue;

    spans[count].begin = clipped_begin - tile_begin;
    spans[count].end = clipped_end - tile_begin;
    spans[count].weight_index = clipped_begin + shift - start;
    ++count;
  }
  return count;
}

// Clip a folded point's support to this block's tile. Returns false when the
// point does not touch the tile on at least one axis.
__host__ __device__ __forceinline__ bool clip_point_to_tile(
    const FoldedPoint &point, TileOrigin origin, Grid grid, SupportSpans &spans) {
  const int end_x = origin.x + TILE_X < grid.nx ? origin.x + TILE_X : grid.nx;
  const int end_y = origin.y + TILE_Y < grid.ny ? origin.y + TILE_Y : grid.ny;
  const int end_z = origin.z + TILE_Z < grid.nz ? origin.z + TILE_Z : grid.nz;

  spans.count_x =
      clip_support_to_tile(point.sx, grid.nx, origin.x, end_x, spans.x);
  spans.count_y =
      clip_support_to_tile(point.sy, grid.ny, origin.y, end_y, spans.y);
  spans.count_z =
      clip_support_to_tile(point.sz, grid.nz, origin.z, end_z, spans.z);
  return spans.count_x && spans.count_y && spans.count_z;
}

// Padded tile-local index. The padding cells are never addressed.
__host__ __device__ __forceinline__ int tile_local_index(int x, int y, int z) {
  return x + PAD_X * (y + TILE_Y * z);
}

__device__ __forceinline__ void clear_tile(float *tile_real, float *tile_imag,
                                           int lane) {
  for (int i = lane; i < TILE_FLOATS; i += THREADS) {
    tile_real[i] = 0.0f;
    tile_imag[i] = 0.0f;
  }
}

// Evaluate the three Horner vectors into this lane's column of shared scratch.
// The layout is [dim * WIDTH + tap][lane], so all lanes in a warp write
// distinct banks. One dimension is live at a time to limit register pressure.
__device__ __forceinline__ void stage_point_weights(
    const FoldedPoint &point, int lane, float (*weights)[THREADS]) {
  using cufinufft::spreadinterp::eval_kernel_vec_horner;
  {
    float ker[WIDTH];
    eval_kernel_vec_horner<float, WIDTH>(ker, float(point.sx) - point.x, 2.0);
    #pragma unroll
    for (int tap = 0; tap < WIDTH; ++tap) weights[tap][lane] = ker[tap];
  }
  {
    float ker[WIDTH];
    eval_kernel_vec_horner<float, WIDTH>(ker, float(point.sy) - point.y, 2.0);
    #pragma unroll
    for (int tap = 0; tap < WIDTH; ++tap)
      weights[WIDTH + tap][lane] = ker[tap];
  }
  {
    float ker[WIDTH];
    eval_kernel_vec_horner<float, WIDTH>(ker, float(point.sz) - point.z, 2.0);
    #pragma unroll
    for (int tap = 0; tap < WIDTH; ++tap)
      weights[2 * WIDTH + tap][lane] = ker[tap];
  }
}

// Add this point's tile-clipped support to the shared tile. Each thread reads
// only its own weight column, so no synchronization is needed inside the point
// loop. The atomicAdd return value is deliberately unused so ptxas can emit a
// fire-and-forget reduction.
__device__ __forceinline__ void scatter_point_support(
    const SupportSpans &spans, float2 strength, int lane,
    const float (*weights)[THREADS], float *tile_real, float *tile_imag) {
  for (int yi = 0; yi < spans.count_y; ++yi) {
    const TileSpan &ys = spans.y[yi];
    for (int y = ys.begin; y < ys.end; ++y) {
      const float wy = weights[WIDTH + ys.weight_index + (y - ys.begin)][lane];
      for (int zi = 0; zi < spans.count_z; ++zi) {
        const TileSpan &zs = spans.z[zi];
        for (int z = zs.begin; z < zs.end; ++z) {
          const float wz =
              weights[2 * WIDTH + zs.weight_index + (z - zs.begin)][lane];
          const float wyz = wy * wz;
          for (int xi = 0; xi < spans.count_x; ++xi) {
            const TileSpan &xs = spans.x[xi];
            for (int x = xs.begin; x < xs.end; ++x) {
              const float wx =
                  weights[xs.weight_index + (x - xs.begin)][lane];
              const float w = wyz * wx;
              const int local = tile_local_index(x, y, z);
              atomicAdd(&tile_real[local], w * strength.x);
              atomicAdd(&tile_imag[local], w * strength.y);
            }
          }
        }
      }
    }
  }
}

// Store every in-grid cell of this tile exactly once, even when the accumulated
// value is zero. The caller poisons the output, so an unconditional store is
// what makes the coverage check meaningful.
__device__ __forceinline__ void write_tile(const float *tile_real,
                                           const float *tile_imag,
                                           TileOrigin origin, Grid grid,
                                           cuFloatComplex *output) {
  for (int i = threadIdx.x; i < TILE_CELLS; i += THREADS) {
    const int x_local = i % TILE_X;
    const int row = i / TILE_X;
    const int y_local = row % TILE_Y;
    const int z_local = row / TILE_Y;
    const int x = origin.x + x_local;
    const int y = origin.y + y_local;
    const int z = origin.z + z_local;
    if (x < grid.nx && y < grid.ny && z < grid.nz) {
      const int local = tile_local_index(x_local, y_local, z_local);
      const long long index =
          x + (long long)grid.nx * (y + (long long)grid.ny * z);
      output[index] = make_cuFloatComplex(tile_real[local], tile_imag[local]);
    }
  }
}

// One block owns one tile; no thread leaves before the final barrier. Points
// are visited bin by bin in the same order as the baseline, but each thread
// scatters its own point immediately, so no per-point block barrier is needed.
__global__ void spread_f0(const Point *points, const int *bin_offsets, Grid grid,
                          cuFloatComplex *output) {
  const TileOrigin origin{static_cast<int>(blockIdx.x) * TILE_X,
                          static_cast<int>(blockIdx.y) * TILE_Y,
                          static_cast<int>(blockIdx.z) * TILE_Z};
  const int lane = threadIdx.x;

  __shared__ detail::CandidateBins x_bins, y_bins, z_bins;
  __shared__ float weights[3 * WIDTH][THREADS];
  __shared__ float tile_real[TILE_FLOATS];
  __shared__ float tile_imag[TILE_FLOATS];

  if (lane == 0) {
    const int end_x = origin.x + TILE_X < grid.nx ? origin.x + TILE_X : grid.nx;
    const int end_y = origin.y + TILE_Y < grid.ny ? origin.y + TILE_Y : grid.ny;
    const int end_z = origin.z + TILE_Z < grid.nz ? origin.z + TILE_Z : grid.nz;
    x_bins = detail::candidate_bins(origin.x, end_x, grid.bin_size, grid.nx,
                                    grid.bins_x);
    y_bins = detail::candidate_bins(origin.y, end_y, grid.bin_size, grid.ny,
                                    grid.bins_y);
    z_bins = detail::candidate_bins(origin.z, end_z, grid.bin_size, grid.nz,
                                    grid.bins_z);
  }
  __syncthreads();

  clear_tile(tile_real, tile_imag, lane);
  __syncthreads();

  for (int z_bin_index = 0; z_bin_index < z_bins.count; ++z_bin_index) {
    const int bz = z_bins[z_bin_index];
    for (int y_bin_index = 0; y_bin_index < y_bins.count; ++y_bin_index) {
      const int by = y_bins[y_bin_index];
      for (int x_bin_index = 0; x_bin_index < x_bins.count; ++x_bin_index) {
        const int bx = x_bins[x_bin_index];
        const int bin = bx + grid.bins_x * (by + grid.bins_y * bz);
        const int bin_begin = bin_offsets[bin];
        const int bin_end = bin_offsets[bin + 1];
        // There are no barriers in this loop, so a thread may skip points whose
        // support misses the tile without synchronizing with its warp or block.
        for (int i = bin_begin + lane; i < bin_end; i += THREADS) {
          const Point point = points[i];
          const FoldedPoint folded = fold_point(point, grid);
          SupportSpans spans;
          if (!clip_point_to_tile(folded, origin, grid, spans)) continue;
          stage_point_weights(folded, lane, weights);
          scatter_point_support(spans,
                                make_float2(point.strength.x, point.strength.y),
                                lane, weights, tile_real, tile_imag);
        }
      }
    }
  }
  __syncthreads();

  write_tile(tile_real, tile_imag, origin, grid, output);
}

}  // namespace f0

void launch_spread_f0(const Point *points, const int *bin_offsets, Grid grid,
                      cuFloatComplex *output) {
  dim3 threads(f0::THREADS);
  dim3 blocks(1 + (grid.nx - 1) / f0::TILE_X,
              1 + (grid.ny - 1) / f0::TILE_Y,
              1 + (grid.nz - 1) / f0::TILE_Z);
  f0::spread_f0<<<blocks, threads>>>(points, bin_offsets, grid, output);
}

}  // namespace atomic_free_spread
