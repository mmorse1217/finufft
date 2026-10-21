#include <cufinufft.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
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

int main(int argc, char **argv) {
  int method = argc > 1 ? std::atoi(argv[1]) : 3;
  int M      = argc > 2 ? std::atoi(argv[2]) : 1000000;
  int N1     = argc > 3 ? std::atoi(argv[3]) : 64;
  int N2     = argc > 4 ? std::atoi(argv[4]) : 64;
  int N3     = argc > 5 ? std::atoi(argv[5]) : 64;
  int nrun   = argc > 6 ? std::atoi(argv[6]) : 10;
  int cluster = argc > 7 ? std::atoi(argv[7]) : 0;
  int binsize = argc > 8 ? std::atoi(argv[8]) : 0;
  int maxsub = argc > 9 ? std::atoi(argv[9]) : 0;
  int np = argc > 10 ? std::atoi(argv[10]) : 0;

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  std::vector<float> x(M), y(M), z(M);
  for (int i = 0; i < M; ++i) {
    if (cluster) {
      // 8 tight Gaussian blobs
      int c = i % 8;
      float cx = (c & 1) ? 0.7f : -0.7f;
      float cy = (c & 2) ? 0.7f : -0.7f;
      float cz = (c & 4) ? 0.7f : -0.7f;
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
  cuFloatComplex *dc, *dfk;
  CU(cudaMalloc(&dx, M * sizeof(float)));
  CU(cudaMalloc(&dy, M * sizeof(float)));
  CU(cudaMalloc(&dz, M * sizeof(float)));
  CU(cudaMalloc(&dc, M * sizeof(cuFloatComplex)));
  long long nf = (long long)N1 * N2 * N3;
  CU(cudaMalloc(&dfk, nf * sizeof(cuFloatComplex)));
  CU(cudaMemcpy(dx, x.data(), M * sizeof(float), cudaMemcpyHostToDevice));
  CU(cudaMemcpy(dy, y.data(), M * sizeof(float), cudaMemcpyHostToDevice));
  CU(cudaMemcpy(dz, z.data(), M * sizeof(float), cudaMemcpyHostToDevice));
  std::vector<cuFloatComplex> c(M, make_cuFloatComplex(0.1f, 0.2f));
  CU(cudaMemcpy(dc, c.data(), M * sizeof(cuFloatComplex), cudaMemcpyHostToDevice));

  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.gpu_method = method;
  opts.gpu_sort = 1;
  opts.gpu_kerevalmeth = 1;
  opts.gpu_spreadinterponly = 1;
  opts.modeord = 0;
  if (binsize) opts.gpu_binsizex = opts.gpu_binsizey = opts.gpu_binsizez = binsize;
  if (maxsub) opts.gpu_maxsubprobsize = maxsub;
  if (np) opts.gpu_np = np;

  int64_t nmodes[3] = {N1, N2, N3};
  cufinufftf_plan plan;

  // Warm-up plan and run.
  nvtxRangePushA("warmup");
  CK(cufinufftf_makeplan(1, 3, nmodes, 1, 1, 1e-5, &plan, &opts));
  CK(cufinufftf_setpts(plan, M, dx, dy, dz, 0, nullptr, nullptr, nullptr));
  CK(cufinufftf_execute(plan, dc, dfk));
  CU(cudaDeviceSynchronize());
  CK(cufinufftf_destroy(plan));
  nvtxRangePop();

  cudaProfilerStart();

  cudaEvent_t e0, e1, e2, e3;
  CU(cudaEventCreate(&e0));
  CU(cudaEventCreate(&e1));
  CU(cudaEventCreate(&e2));
  CU(cudaEventCreate(&e3));

  double best_setpts = 1e9, best_exec = 1e9, best_total = 1e9;
  for (int r = 0; r < nrun; ++r) {
    nvtxRangePushA("iteration");
    nvtxRangePushA("makeplan");
    CK(cufinufftf_makeplan(1, 3, nmodes, 1, 1, 1e-5, &plan, &opts));
    nvtxRangePop(); // makeplan
    CU(cudaEventRecord(e0));
    nvtxRangePushA("setpts");
    CK(cufinufftf_setpts(plan, M, dx, dy, dz, 0, nullptr, nullptr, nullptr));
    CU(cudaEventRecord(e1));
    nvtxRangePop(); // setpts
    nvtxRangePushA("execute");
    CK(cufinufftf_execute(plan, dc, dfk));
    CU(cudaEventRecord(e2));
    CU(cudaEventSynchronize(e2));
    nvtxRangePop(); // execute
    nvtxRangePop(); // iteration
    float tsp = 0, tex = 0, ttot = 0;
    CU(cudaEventElapsedTime(&tsp, e0, e1));
    CU(cudaEventElapsedTime(&tex, e1, e2));
    CU(cudaEventElapsedTime(&ttot, e0, e2));
    if (tsp < best_setpts) best_setpts = tsp;
    if (tex < best_exec) best_exec = tex;
    if (ttot < best_total) best_total = ttot;
    CK(cufinufftf_destroy(plan));
  }

  cudaProfilerStop();

  std::printf("method=%d M=%d nf=%lld cluster=%d bin=%d maxsub=%d np=%d setpts=%.3f ms execute=%.3f ms total=%.3f ms  Mpts/s(exec)=%.2f\n",
              method, M, nf, cluster, binsize, maxsub, np, best_setpts, best_exec, best_total,
              M / best_exec / 1000.0);
  return 0;
}
