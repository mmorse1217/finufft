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
static constexpr int NS      = 6;  // kernel width, ns=6 for tol 1e-5 at sigma=2
static constexpr int TILE_X  = 16;
static constexpr int TILE_Y  = 16;
static constexpr int TILE_Z  = 8;
static constexpr int NTHREAD = 128;                 // 4 warps, 8x8x8 warp region
static constexpr int NWARPS  = NTHREAD / 32;
static constexpr int HALO    = (NS + 1) / 2;        // 3
static constexpr int GSX     = 2;                   // cells per thread per dim
static constexpr int GSY     = 2;
static constexpr int GSZ     = 4;                   // 16 cells per thread

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
  const int nwarps = tiles_x * tiles_y * ((nf3 + TILE_Z - 1) / TILE_Z) * NWARPS;
  const int wid    = blockIdx.x * blockDim.x + threadIdx.x;
  if (wid >= nwarps) return;
  const int warp = wid % NWARPS;
  const int tile = wid / NWARPS;
  const int tz   = tile / (tiles_x * tiles_y);
  const int ty   = (tile / tiles_x) % tiles_y;
  const int tx   = tile % tiles_x;
  const int X0   = tx * TILE_X + (warp & 1) * 8;
  const int Y0   = ty * TILE_Y + ((warp >> 1) & 1) * 8;
  const int Z0   = tz * TILE_Z + (warp >> 2) * 8;
  const int bx0 = floordiv(X0 - HALO, qb), bx1 = (X0 + 7 + HALO) / qb;
  const int by0 = floordiv(Y0 - HALO, qb), by1 = (Y0 + 7 + HALO) / qb;
  const int bz0 = floordiv(Z0 - HALO, qb), bz1 = (Z0 + 7 + HALO) / qb;
  unsigned long long v = 0;
  for (int ix = bx0; ix <= bx1; ix++) {
    int wx = ix % nqx;
    wx += (wx >> 31) & nqx;
    for (int iy = by0; iy <= by1; iy++) {
      int wy = iy % nqy;
      wy += (wy >> 31) & nqy;
      for (int iz = bz0; iz <= bz1; iz++) {
        int wz = iz % nqz;
        wz += (wz >> 31) & nqz;
        const int bin = wx + nqx * (wy + nqy * wz);
        v += (unsigned long long)(binstart[bin + 1] - binstart[bin]);
      }
    }
  }
  // No atomics, no shared reduction: one slot per warp, summed on the host.
  visits[wid]       = v;
  active_warps[wid] = (v != 0) ? 1ULL : 0ULL;
}

// ---------------------------------------------------------------------------
// Output-cell-owned spread kernel (warp-cooperative gather).
//
// Block = 256 threads = 8 warps, one tile (16x16x8) per block.
// Warp region = 8x8x4 cells; lane -> 2x2x2 group (lane grid 4x4x2).
// The warp enumerates the query bins covering its region expanded by HALO and
// stages each point cooperatively: 18 lanes evaluate one Horner coefficient
// each, one lane records the support start, one lane records the strength.
// Staged points are in shared memory, so every lane reads the same bytes
// (broadcast) and can index the kernel values dynamically at no local-memory
// cost. Each lane accumulates its own 8 cells and writes them exactly once.
// ---------------------------------------------------------------------------
__device__ __constant__ float KER6[9][6] = {
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

static constexpr int BATCH = 16;
static constexpr int STAGE = 24;  // 18 kernel + 3 coord/support + 2 strength + pad

// Raw pass: lanes 0..cnt-1 issue all the global loads for the batch at once, so
// the warp waits on one round of coalesced loads instead of ~cnt serialized ones.
// Coordinates and strength land in slots 18..22 of each point's stage row.
__device__ __forceinline__ void stage_raw(int q0, int cnt, int lane,
                                          const float *__restrict__ x,
                                          const float *__restrict__ y,
                                          const float *__restrict__ z,
                                          const float2 *__restrict__ cstr,
                                          float *__restrict__ buf) {
  if (lane < cnt) {
    const int q      = q0 + lane;
    float *const row = buf + lane * STAGE;
    row[18]          = __ldg(x + q);
    row[19]          = __ldg(y + q);
    row[20]          = __ldg(z + q);
    const float2 cp  = __ldg(cstr + q);
    row[21]          = cp.x;
    row[22]          = cp.y;
  }
}

// Coefficient pass for one staged point: lanes 0-17 evaluate one Horner coefficient
// each (nine polynomial coefficients preloaded in registers); the six lanes of a
// dimension fold the same broadcast coordinate in parallel, and the support start is
// written by all six lanes of that dimension (same value, so no branch is needed).
__device__ __forceinline__ void stage_coeff(float *__restrict__ row, int lane,
                                            int nf1, int nf2, int nf3,
                                            const float *__restrict__ coeff) {
  using cufinufft::spreadinterp::fold_rescale;
  if (lane < 18) {
    const int d   = lane / 6;
    const int nfd = (d == 0) ? nf1 : ((d == 1) ? nf2 : nf3);
    const float r    = fold_rescale(row[18 + d], nfd);
    const float sval = (float)((int)ceilf(r - (float)(NS / 2)));
    const float zz   = __fmaf_rn(2.f, sval - r, (float)(NS - 1));
    row[18 + d] = sval;
    float v     = coeff[8];
#pragma unroll
    for (int t = 7; t >= 0; t--) v = __fmaf_rn(zz, v, coeff[t]);
    row[lane] = v;
  }
}

// Support placement of a point in the frame of a GSIZE-cell group starting at g0.
// Branchless: n = number of group cells covered (0,1,2); the covered cells are
// contiguous, local cell l0+k gets kernel coefficient k[j0+k] for k < n.
__device__ __forceinline__ void dim_support(int s, int nfd, int g0, int gs, int &n,
                                            int &j0, int &l0) {
  int d = s - g0;
  d += nfd & (d >> 31);       // d < 0    -> d + nfd
  d -= nfd & -(d >= nfd);     // d >= nfd -> d - nfd
  // n/j0/l0 below are selects: ptxas lowers them to SEL/ISETP with no branch
  // (SASS branch count is identical to an explicit mask form, which measured
  // 11-17% slower because it adds ~21 LOP3 per scan).
  const bool low  = d < gs;
  const bool high = d >= nfd - (NS - 1);
  const int nh    = NS - (nfd - d);
  n  = low ? (gs - d) : (high ? (nh < gs ? nh : gs) : 0);
  j0 = low ? 0 : (high ? (nfd - d) : 0);
  l0 = low ? d : 0;
}

// One staged point: support placement, kernel values, 16-cell accumulation.
// Pulled out of the loop so the scan can issue two points back to back (ILP).
__device__ __forceinline__ void scan_point(const float *__restrict__ row, int nf1, int nf2,
                                           int nf3, int gx0, int gy0, int gz0,
                                           float2 (&acc)[16]) {
  const float2 s01 = *reinterpret_cast<const float2 *>(row + 18);  // s x,y
  const float2 sc  = *reinterpret_cast<const float2 *>(row + 20);  // s z, strength re
  const int sx = (int)s01.x, sy = (int)s01.y, sz = (int)sc.x;
  const float cx = sc.y, cy = row[22];
  int nx, jx, lx, ny, jy, ly, nz, jz, lz;
  dim_support(sx, nf1, gx0, GSX, nx, jx, lx);
  dim_support(sy, nf2, gy0, GSY, ny, jy, ly);
  dim_support(sz, nf3, gz0, GSZ, nz, jz, lz);
  if (!(nx && ny && nz)) return;
  const float kx0 = row[jx], kx1 = (nx >= 2) ? row[jx + 1] : 0.f;
  const float ky0 = row[6 + jy], ky1 = (ny >= 2) ? row[7 + jy] : 0.f;
  const float kz0 = row[12 + jz], kz1 = (nz >= 2) ? row[13 + jz] : 0.f;
  const float kz2 = (nz >= 3) ? row[14 + jz] : 0.f;
  const float kz3 = (nz >= 4) ? row[15 + jz] : 0.f;
  const float vx[2] = {(lx == 0) ? kx0 : 0.f, (lx == 0) ? kx1 : kx0};
  const float vy[2] = {(ly == 0) ? ky0 : 0.f, (ly == 0) ? ky1 : ky0};
  const float vz[4] = {(lz == 0) ? kz0 : 0.f,
                       (lz == 0) ? kz1 : ((lz == 1) ? kz0 : 0.f),
                       (lz == 0) ? kz2 : ((lz == 1) ? kz1 : ((lz == 2) ? kz0 : 0.f)),
                       (lz == 0) ? kz3 : ((lz == 1) ? kz2
                                          : ((lz == 2) ? kz1 : ((lz == 3) ? kz0 : 0.f)))};
  // Indexed by cx + 2*cy to match the accumulator layout.
  const float wxy[4] = {vx[0] * vy[0], vx[1] * vy[0], vx[0] * vy[1], vx[1] * vy[1]};
#pragma unroll
  for (int cz = 0; cz < 4; cz++) {
#pragma unroll
    for (int cy2 = 0; cy2 < 2; cy2++) {
#pragma unroll
      for (int cx2 = 0; cx2 < 2; cx2++) {
        const int li    = cx2 + 2 * cy2 + 4 * cz;
        const float wgt = wxy[cx2 + 2 * cy2] * vz[cz];
        acc[li].x += cx * wgt;
        acc[li].y += cy * wgt;
      }
    }
  }
}

__global__ void spread_cartesian_owned(
    const float *__restrict__ x, const float *__restrict__ y, const float *__restrict__ z,
    const cuFloatComplex *__restrict__ c, const int *__restrict__ binstart,
    cuFloatComplex *__restrict__ fw, int nf1, int nf2,
    int nf3, int nqx, int nqy, int nqz, int qb, int tiles_x, int tiles_y, int nsplit) {
  __shared__ __align__(16) float stage[NWARPS][2][BATCH][STAGE];

  const int warp  = threadIdx.x >> 5;
  const int lane  = threadIdx.x & 31;
  const int tile  = blockIdx.x / nsplit;
  const int split = blockIdx.x - tile * nsplit;
  const int tz   = tile / (tiles_x * tiles_y);
  const int ty   = (tile / tiles_x) % tiles_y;
  const int tx   = tile % tiles_x;
  const int X0   = tx * TILE_X + (warp & 1) * 8;
  const int Y0   = ty * TILE_Y + ((warp >> 1) & 1) * 8;
  const int Z0   = tz * TILE_Z + (warp >> 2) * 8;
  const int gx0  = X0 + (lane & 3) * GSX;
  const int gy0  = Y0 + ((lane >> 2) & 3) * GSY;
  const int gz0  = Z0 + ((lane >> 4) & 1) * GSZ;
  // No early return: __syncwarp() must be executed by the whole warp. Groups
  // outside the grid (partial edge tiles) accumulate garbage but never store.

  float2 acc[16];
#pragma unroll
  for (int i = 0; i < 16; i++) acc[i] = make_float2(0.f, 0.f);

  // The nine Horner coefficients for this lane's index are loop-invariant: load
  // them once into registers instead of doing a divergent constant load per point.
  float coeff[9] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  if (lane < 18) {
    const int ci = lane % 6;
#pragma unroll
    for (int t = 0; t < 9; t++) coeff[t] = KER6[t][ci];
  }

  const int bx0 = floordiv(X0 - HALO, qb), bx1 = (X0 + 7 + HALO) / qb;
  const int by0 = floordiv(Y0 - HALO, qb), by1 = (Y0 + 7 + HALO) / qb;
  const int bz0 = floordiv(Z0 - HALO, qb), bz1 = (Z0 + 7 + HALO) / qb;
  const float2 *const cf = reinterpret_cast<const float2 *>(c);

  // Walk the warp's bin cube as contiguous x-runs. Bins along x are consecutive in the
  // sorted point table, so a whole x-run is one contiguous point range and can be batched
  // in 16s regardless of the query-bin size (this is what makes qb=1, reread 5.36, usable).
  const int nxbin = min(bx1 - bx0 + 1, nqx);
  int wlo         = bx0 % nqx;
  wlo += (wlo >> 31) & nqx;
  const int wend = wlo + nxbin - 1;
  for (int iy = by0; iy <= by1; iy++) {
    int wy = iy % nqy;
    wy += (wy >> 31) & nqy;
    for (int iz = bz0; iz <= bz1; iz++) {
      int wz = iz % nqz;
      wz += (wz >> 31) & nqz;
      const int base = nqx * (wy + nqy * wz);
      for (int seg = 0; seg < 2; ++seg) {
        // The x-run can wrap around the periodic boundary: two contiguous segments.
        const int x0 = (seg == 0) ? wlo : 0;
        const int x1 = (seg == 0) ? min(wend, nqx - 1) : (wend - nqx);
        if (x1 < x0) continue;
        const int qlo  = binstart[base + x0];
        const int qhi  = binstart[base + x1 + 1];
        const int step = nsplit * BATCH;
        int q0         = qlo + split * BATCH;
        if (q0 < qhi) {
          int cnt    = min(BATCH, qhi - q0);
          float *cur = &stage[warp][0][0][0];
          float *nxt = &stage[warp][1][0][0];
          stage_raw(q0, cnt, lane, x, y, z, cf, cur);
          __syncwarp();
          for (;;) {
            for (int b = 0; b < cnt; b++)
              stage_coeff(cur + b * STAGE, lane, nf1, nf2, nf3, coeff);
            __syncwarp();
            const int qn   = q0 + step;
            const int cntn = (qn < qhi) ? min(BATCH, qhi - qn) : 0;
            if (cntn) stage_raw(qn, cntn, lane, x, y, z, cf, nxt);
            int b = 0;
            for (; b + 1 < cnt; b += 2) {
              scan_point(cur + b * STAGE, nf1, nf2, nf3, gx0, gy0, gz0, acc);
              scan_point(cur + (b + 1) * STAGE, nf1, nf2, nf3, gx0, gy0, gz0, acc);
            }
            if (b < cnt) scan_point(cur + b * STAGE, nf1, nf2, nf3, gx0, gy0, gz0, acc);
            __syncwarp();
            if (!cntn) break;
            q0  = qn;
            cnt = cntn;
            float *tmp = cur;
            cur        = nxt;
            nxt        = tmp;
          }
        }
      }
    }
  }

  // Unique, non-atomic store into this split's slice (split 0 only when nsplit==1).
  const long long nf_total = (long long)nf1 * nf2 * nf3;
  cuFloatComplex *const fw_out = fw + (long long)split * nf_total;
#pragma unroll
  for (int i = 0; i < 16; i++) {
    const int lx = i & 1, ly = (i >> 1) & 1, lz = (i >> 2) & 3;
    const long long gx = gx0 + lx, gy = gy0 + ly, gz = gz0 + lz;
    if (gx < nf1 && gy < nf2 && gz < nf3) {
      const long long idx = gx + (long long)nf1 * (gy + (long long)nf2 * gz);
      fw_out[idx]         = make_cuFloatComplex(acc[i].x, acc[i].y);
    }
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
  int *d_perm = nullptr, *d_starts = nullptr;
  void *d_sort_tmp = nullptr;
  size_t sort_tmp_bytes = 0;

  CU(cudaMalloc(&d_qbin, M * sizeof(int)));
  CU(cudaMalloc(&d_counts, nbins * sizeof(int)));
  CU(cudaMalloc(&d_cursor, nbins * sizeof(int)));
  CU(cudaMalloc(&d_binstart, (nbins + 1) * sizeof(int)));
  CU(cudaMalloc(&d_sorted, M * sizeof(int)));
  CU(cudaMalloc(&d_sx, M * sizeof(float)));
  CU(cudaMalloc(&d_sy, M * sizeof(float)));
  CU(cudaMalloc(&d_sz, M * sizeof(float)));
  CU(cudaMalloc(&d_sc, M * sizeof(cuFloatComplex)));
  CU(cudaMalloc(&d_keys_out, M * sizeof(unsigned int)));
  CU(cudaMalloc(&d_perm, M * sizeof(int)));
  CU(cudaMalloc(&d_starts, nbins * sizeof(int)));
  if (prep == 1) {
    size_t bytes = 0;
    cub::DeviceRadixSort::SortPairs(nullptr, bytes, (const unsigned int *)d_qbin, d_keys_out,
                                    (const unsigned int *)d_sorted, (unsigned int *)d_perm, M,
                                    0, end_bit);
    sort_tmp_bytes = bytes;
    CU(cudaMalloc(&d_sort_tmp, sort_tmp_bytes));
  }
  std::vector<int> hstart(nbins + 1, 0);
  std::vector<int> hstarts(nbins, 0);
  std::vector<cuFloatComplex> h_ref(nf), h_new(nf);

  std::vector<double> pre_ms;
  for (int r = -2; r < nrun; r++) {
    CU(cudaDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    nvtxRangePushA("preprocess_qbin");
    kernel_compute_qbin<<<(M + 255) / 256, 256>>>(dx, dy, dz, M, N1, N2, N3, qb, nqx, nqy,
                                                  nqz, d_qbin);
    nvtxRangePop();
    if (prep == 1) {
      // CUB radix sort of (bin, point index): no histogram, no per-bin atomics.
      nvtxRangePushA("preprocess_sort");
      kernel_fill_idx<<<(M + 255) / 256, 256>>>(d_sorted, M);
      cub::DeviceRadixSort::SortPairs(d_sort_tmp, sort_tmp_bytes, (const unsigned int *)d_qbin,
                                      d_keys_out, (const unsigned int *)d_sorted,
                                      (unsigned int *)d_perm, M, 0, end_bit);
      kernel_gather_sorted<<<(M + 255) / 256, 256>>>(d_perm, dx, dy, dz, dc, d_sx, d_sy, d_sz,
                                                     d_sc, M);
      nvtxRangePop();
      nvtxRangePushA("preprocess_starts");
      CU(cudaMemset(d_starts, 0xFF, nbins * sizeof(int)));
      kernel_mark_starts<<<(M + 255) / 256, 256>>>(d_keys_out, M, d_starts);
      CU(cudaMemcpy(hstarts.data(), d_starts, nbins * sizeof(int), cudaMemcpyDeviceToHost));
      hstart[nbins] = M;
      for (int b = nbins - 1; b >= 0; --b)
        hstart[b] = (hstarts[b] >= 0 && hstarts[b] < hstart[b + 1]) ? hstarts[b] : hstart[b + 1];
      CU(cudaMemcpy(d_binstart, hstart.data(), (nbins + 1) * sizeof(int),
                    cudaMemcpyHostToDevice));
      nvtxRangePop();
    } else {
      // Counting sort: atomic histogram, host scan, atomic per-bin scatter.
      CU(cudaMemset(d_counts, 0, nbins * sizeof(int)));
      nvtxRangePushA("preprocess_hist");
      kernel_hist<<<(M + 255) / 256, 256>>>(d_qbin, M, d_counts);
      nvtxRangePop();
      nvtxRangePushA("preprocess_scan");
      std::vector<int> hcount(nbins);
      CU(cudaMemcpy(hcount.data(), d_counts, nbins * sizeof(int), cudaMemcpyDeviceToHost));
      int acc = 0;
      for (int b = 0; b < nbins; b++) {
        hstart[b] = acc;
        acc += hcount[b];
      }
      hstart[nbins] = acc;
      CU(cudaMemcpy(d_binstart, hstart.data(), (nbins + 1) * sizeof(int),
                    cudaMemcpyHostToDevice));
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
  const int tiles_x = (N1 + TILE_X - 1) / TILE_X;
  const int tiles_y = (N2 + TILE_Y - 1) / TILE_Y;
  const int tiles_z = (N3 + TILE_Z - 1) / TILE_Z;
  const int ntiles  = tiles_x * tiles_y * tiles_z;

  // ---- candidate visits / active warps (untimed) and split decision ----
  const int nwarps_total = ntiles * NWARPS;
  unsigned long long *d_visits, *d_active;
  CU(cudaMalloc(&d_visits, nwarps_total * sizeof(unsigned long long)));
  CU(cudaMalloc(&d_active, nwarps_total * sizeof(unsigned long long)));
  kernel_visits<<<(nwarps_total + 255) / 256, 256>>>(d_binstart, nqx, nqy, nqz, N1, N2, N3,
                                                     qb, tiles_x, tiles_y, d_visits, d_active);
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
  // Split only when few warp regions are active (clustered input): raise the number of
  // work items until ~256 warps are in flight. Each split owns a subset of the bins and
  // writes its own partial grid; the reduction sums them.
  int nsplit = 1;
  while (nsplit < 8 && (long long)h_active * nsplit < 512) nsplit <<= 1;
  cuFloatComplex *dfk_part = nullptr;
  if (nsplit > 1) CU(cudaMalloc(&dfk_part, (long long)nsplit * nf * sizeof(cuFloatComplex)));

  std::vector<double> new_ms;
  cudaEvent_t e0, e1;
  CU(cudaEventCreate(&e0));
  CU(cudaEventCreate(&e1));
  for (int r = -2; r < nrun; r++) {
    CU(cudaEventRecord(e0));
    nvtxRangePushA(nsplit > 1 ? "spread_cartesian_owned_split" : "spread_cartesian_owned");
    spread_cartesian_owned<<<ntiles * nsplit, NTHREAD>>>(
        d_sx, d_sy, d_sz, d_sc, d_binstart, nsplit > 1 ? dfk_part : dfk_new, N1, N2, N3, nqx,
        nqy, nqz, qb, tiles_x, tiles_y, nsplit);
    if (nsplit > 1)
      kernel_sum_splits<<<(int)((nf + 255) / 256), 256>>>(dfk_part, nsplit, nf, dfk_new);
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
  CU(cudaFuncGetAttributes(&attr, spread_cartesian_owned));
  int blocks_per_sm = 0;
  CU(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, spread_cartesian_owned,
                                                   NTHREAD, 0));

  std::printf("case=M%d nf=%dx%dx%d cluster=%d qb=%d\n", M, N1, N2, N3, cluster, qb);
  std::printf("ref   method=3 setpts=%.3f ms exec=%.3f ms total=%.3f ms\n", t3.setpts_ms,
              t3.exec_ms, t3.total_ms);
  std::printf("ctx   method=4 setpts=%.3f ms exec=%.3f ms total=%.3f ms\n", t4.setpts_ms,
              t4.exec_ms, t4.total_ms);
  std::printf("ctx   method=1 setpts=%.3f ms exec=%.3f ms total=%.3f ms\n", t1.setpts_ms,
              t1.exec_ms, t1.total_ms);
  std::printf("new   qbins=%d^3 bins=%d prep=%s preprocess=%.3f ms spread=%.3f ms total=%.3f ms nsplit=%d\n",
              nqx, nbins, prep == 0 ? "count" : "radix", preprocess_ms, new_ms_med,
              preprocess_ms + new_ms_med, nsplit);
  std::printf("new   speedup_vs_m3_spread=%.2fx spread_ratio=%.3f total_ratio=%.3f\n",
              t3.exec_ms / new_ms_med, new_ms_med / t3.exec_ms,
              (preprocess_ms + new_ms_med) / t3.total_ms);
  std::printf("new   relL2=%.3e maxabs=%.3e visits=%llu reread_factor=%.2f active_warps=%llu/%d regs=%d smem=%zu occ_blocks_per_sm=%d\n",
              relL2, maxabs, h_visits, (double)h_visits / (double)M, h_active, nwarps_total,
              attr.numRegs, attr.sharedSizeBytes, blocks_per_sm);
  return 0;
}
