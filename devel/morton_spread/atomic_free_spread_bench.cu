// Benchmark and correctness harness for the readable, atomic-free 3D type-1
// spreading baseline in atomic_free_spread.cu.
//
// It reports per-phase timings (host preparation, GPU bin assignment, median
// spread-kernel time, and the pinned cuFINUFFT method 3 reference) and checks
// the result against method 3 plus independent structural invariants: valid bin
// IDs, a monotonic bin-offset table, every output cell written, and finite
// output. The process exits non-zero if any check fails.
//
// Build and run with ./devel/morton_spread/run_atomic_free_spread.sh.
//
// Usage: atomic_free_spread_bench
//            [M [N1 [N2 [N3 [cluster [bin_size [pattern [seed [nrun
//            [kernel [bin_sort]]]]]]]]]]]
// cluster: 0 = uniform points, 1 = eight tight clusters.
// pattern: 0 = uniform/cluster (default), 1 = seam and out-of-domain points,
//          2 = exact grid nodes and near-node offsets, 3 = one dense cluster.
//          Patterns 1-3 ignore cluster and use seed for reproducible selection.
// nrun: paired timed runs per scope after two warm-ups (default 5; median reported).
// kernel: 0 = atomic-free baseline (default), 1 = F0 point-owned spread.
// bin_sort: 0 = CPU fallback, 1 = stable radix GPU bin sort (default).

#include "atomic_free_spread.cuh"
#include "gpu_bin_sort.cuh"

#include <cuda_runtime.h>
#include <cufinufft.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call) do {                                                        \
  cudaError_t error = (call);                                                        \
  if (error != cudaSuccess) {                                                        \
    std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call,              \
                 cudaGetErrorString(error));                                         \
    std::exit(1);                                                                    \
  }                                                                                  \
} while (0)

#define FINUFFT_CHECK(call) do {                                                     \
  int error = (call);                                                                \
  if (error != 0) {                                                                  \
    std::fprintf(stderr, "%s:%d: %s failed: %d\n", __FILE__, __LINE__, #call, error); \
    std::exit(1);                                                                    \
  }                                                                                  \
} while (0)

namespace {

using atomic_free_spread::Grid;
using atomic_free_spread::Point;
using Clock = std::chrono::steady_clock;

constexpr float PI = 3.14159265358979323846f;
constexpr double REL_L2_TOLERANCE = 1e-4;
constexpr double TOTAL_TOLERANCE = 1e-4;

double elapsed_ms(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// CUDA-event stopwatch. end() synchronizes, so the returned interval includes
// all work queued on the default stream between begin() and end().
struct CudaTimer {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  CudaTimer() {
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
  }
  ~CudaTimer() {
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
  }
  CudaTimer(const CudaTimer &) = delete;
  CudaTimer &operator=(const CudaTimer &) = delete;

  void begin() { CUDA_CHECK(cudaEventRecord(start)); }
  double end() {
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    return milliseconds;
  }
};

template<typename T>
class DeviceBuffer {
 public:
  explicit DeviceBuffer(size_t count) {
    CUDA_CHECK(cudaMalloc(&data_, count * sizeof(T)));
  }
  ~DeviceBuffer() {
    if (data_) cudaFree(data_);
  }
  DeviceBuffer(const DeviceBuffer &) = delete;
  DeviceBuffer &operator=(const DeviceBuffer &) = delete;

  T *get() const { return data_; }

 private:
  T *data_ = nullptr;
};

struct HostPoints {
  std::vector<Point> points;
  std::vector<float> x, y, z;
  std::vector<cuFloatComplex> strengths;
};

HostPoints generate_points(int count, int nx, int ny, int nz, int cluster,
                           int pattern, int seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
  std::normal_distribution<float> normal(0.0f, 0.01f);
  std::normal_distribution<float> narrow(0.0f, 1e-4f);
  HostPoints host;
  host.points.resize(count);
  host.x.resize(count);
  host.y.resize(count);
  host.z.resize(count);
  host.strengths.resize(count);
  // Build a coordinate near one grid node. The float cast is deliberate: the
  // device fold has to decide which cell the rounded value lands in.
  auto grid_node_coordinate = [&](int i, int n, int axis) {
    const int node = (int)((unsigned)(i * 2654435761u + axis * 40503u + seed) %
                           (unsigned)n);
    const int variant = (int)((unsigned)(i * 1103515245u + axis * 12345u + seed) % 20u);
    const double node_coordinate = 2.0 * 3.14159265358979323846 *
                                   ((double)node / (double)n - 0.5);
    if (variant == 1) return (float)node_coordinate;                 // exact node
    if (variant == 2) return (float)(node_coordinate - 1e-7);        // just below
    if (variant == 3) return (float)(node_coordinate + 1e-7);        // just above
    if (variant == 0)
      return (float)(node_coordinate + 1e-7 * (i % 3 - 1));          // near node
    return (float)(node_coordinate + 0.37 / (double)n);              // inside the cell
  };
  for (int i = 0; i < count; ++i) {
    if (pattern == 0 && cluster) {
      int c = i % 8;
      host.x[i] = PI * ((c & 1 ? 0.7f : -0.7f) + normal(rng));
      host.y[i] = PI * ((c & 2 ? 0.7f : -0.7f) + normal(rng));
      host.z[i] = PI * ((c & 4 ? 0.7f : -0.7f) + normal(rng));
    } else if (pattern == 0) {
      host.x[i] = PI * uniform(rng);
      host.y[i] = PI * uniform(rng);
      host.z[i] = PI * uniform(rng);
    } else if (pattern == 1) {
      // Exact folds, both sides of the seam, and out-of-domain coordinates.
      static const float seam_coordinates[] = {
          -7.0f * PI, -3.0f * PI, -2.0f * PI, -PI - 1e-6f, -PI, -PI + 1e-6f,
          -1e-6f, 0.0f, 1e-6f, PI - 1e-6f, PI, PI + 1e-6f,
          2.0f * PI, 3.0f * PI, 7.0f * PI};
      const int coordinate_count =
          (int)(sizeof(seam_coordinates) / sizeof(seam_coordinates[0]));
      host.x[i] = seam_coordinates[(i + 3 * seed) % coordinate_count];
      host.y[i] = seam_coordinates[(i * 5 + 7 + seed) % coordinate_count];
      host.z[i] = seam_coordinates[(i * 11 + 13 + seed) % coordinate_count];
    } else if (pattern == 2) {
      host.x[i] = grid_node_coordinate(i, nx, 1);
      host.y[i] = grid_node_coordinate(i, ny, 2);
      host.z[i] = grid_node_coordinate(i, nz, 3);
    } else {
      // One dense cluster: many points share a cell, so summation order matters.
      host.x[i] = 0.123f + narrow(rng);
      host.y[i] = -0.456f + narrow(rng);
      host.z[i] = 0.789f + narrow(rng);
    }
    // Vary both components so correctness does not depend on identical strengths.
    host.strengths[i] = make_cuFloatComplex(uniform(rng), uniform(rng));
    host.points[i] = {host.x[i], host.y[i], host.z[i], host.strengths[i]};
  }
  return host;
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

struct TimingPair {
  std::vector<double> ours, method3;
};

// Equal warmups and samples, reversing order each round. CUDA events measure
// device-resident work; synchronized wall time includes host transfers/work.
template<typename Ours, typename Method3>
TimingPair measure_pair(int nrun, bool wall_time, Ours ours, Method3 method3) {
  TimingPair samples;
  CudaTimer timer;
  auto measure = [&](auto &operation) {
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = Clock::now();
    if (!wall_time) timer.begin();
    operation();
    CUDA_CHECK(cudaGetLastError());
    if (!wall_time) return timer.end();
    CUDA_CHECK(cudaDeviceSynchronize());
    return elapsed_ms(start);
  };
  for (int run = -2; run < nrun; ++run) {
    double a, b;
    if (run % 2 == 0) {
      a = measure(ours);
      b = measure(method3);
    } else {
      b = measure(method3);
      a = measure(ours);
    }
    if (run >= 0) {
      samples.ours.push_back(a);
      samples.method3.push_back(b);
    }
  }
  return samples;
}

struct ReferenceResult {
  std::vector<cuFloatComplex> grid;
  double setpts_ms = 0.0;
  double execute_ms = 0.0;
  double total_ms = 0.0;
  TimingPair spread, gpu_pipeline, host_pipeline;
};

// Both implementations reuse allocations/plans. Generation, validation, and
// allocation are outside every timing scope. Host inputs use native layouts.
template<typename Upload, typename Prepare, typename Spread, typename Download>
ReferenceResult run_reference(const HostPoints &host, int nx, int ny, int nz,
                              long long num_cells, int nrun, bool gpu_sort,
                              Upload upload, Prepare prepare, Spread spread,
                              Download download) {
  const int count = (int)host.points.size();
  DeviceBuffer<float> dx(count), dy(count), dz(count);
  DeviceBuffer<cuFloatComplex> dc(count), reference_grid(num_cells);
  const auto upload_reference = [&]() {
    CUDA_CHECK(cudaMemcpy(dx.get(), host.x.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy.get(), host.y.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dz.get(), host.z.data(), count * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dc.get(), host.strengths.data(), count * sizeof(cuFloatComplex),
                          cudaMemcpyHostToDevice));
  };
  upload_reference();

  cufinufft_opts opts;
  cufinufft_default_opts(&opts);
  opts.gpu_method = 3;
  opts.gpu_sort = 1;
  opts.gpu_kerevalmeth = 1;
  opts.gpu_spreadinterponly = 1;
  opts.modeord = 0;
  int64_t dims[3] = {nx, ny, nz};
  cufinufftf_plan plan;
  FINUFFT_CHECK(cufinufftf_makeplan(1, 3, dims, 1, 1, 1e-5, &plan, &opts));

  ReferenceResult result;
  result.grid.resize((size_t)num_cells);
  const auto setpts = [&]() {
    FINUFFT_CHECK(cufinufftf_setpts(plan, count, dx.get(), dy.get(), dz.get(), 0,
                                   nullptr, nullptr, nullptr));
  };
  const auto execute = [&]() {
    FINUFFT_CHECK(cufinufftf_execute(plan, dc.get(), reference_grid.get()));
  };
  const auto download_reference = [&]() {
    CUDA_CHECK(cudaMemcpy(result.grid.data(), reference_grid.get(),
                          (size_t)num_cells * sizeof(cuFloatComplex),
                          cudaMemcpyDeviceToHost));
  };

  // Diagnostic setpts median; totals below are measured as whole pipelines,
  // never constructed by adding independently measured phase medians.
  CudaTimer timer;
  std::vector<double> setpts_samples;
  for (int run = -2; run < nrun; ++run) {
    timer.begin();
    setpts();
    const double ms = timer.end();
    if (run >= 0) setpts_samples.push_back(ms);
  }
  result.setpts_ms = median(setpts_samples);
  result.spread = measure_pair(nrun, false, spread, execute);
  result.execute_ms = median(result.spread.method3);
  if (gpu_sort) {
    result.gpu_pipeline = measure_pair(nrun, false,
        [&]() { prepare(); spread(); }, [&]() { setpts(); execute(); });
    result.total_ms = median(result.gpu_pipeline.method3);
  }
  result.host_pipeline = measure_pair(nrun, true,
      [&]() { upload(); prepare(); spread(); download(); },
      [&]() { upload_reference(); setpts(); execute(); download_reference(); });

  // Fresh, untimed correctness run: previous outputs cannot mask missing writes.
  CUDA_CHECK(cudaMemset(reference_grid.get(), 0xFF,
                        (size_t)num_cells * sizeof(cuFloatComplex)));
  setpts();
  execute();
  download_reference();
  FINUFFT_CHECK(cufinufftf_destroy(plan));
  return result;
}

bool is_finite(const cuFloatComplex &value) {
  return std::isfinite(value.x) && std::isfinite(value.y);
}

// cudaMemset(0xFF) makes both components 0xFFFFFFFF, an IEEE NaN pattern that
// the spread kernel never produces for finite input.
bool is_unwritten_sentinel(const cuFloatComplex &value) {
  std::uint32_t real_bits = 0, imag_bits = 0;
  std::memcpy(&real_bits, &value.x, sizeof(real_bits));
  std::memcpy(&imag_bits, &value.y, sizeof(imag_bits));
  return real_bits == 0xFFFFFFFFu && imag_bits == 0xFFFFFFFFu;
}

const char *pass_fail(bool pass) { return pass ? "PASS" : "FAIL"; }

bool same_float_bits(float a, float b) {
  std::uint32_t a_bits = 0, b_bits = 0;
  std::memcpy(&a_bits, &a, sizeof(a_bits));
  std::memcpy(&b_bits, &b, sizeof(b_bits));
  return a_bits == b_bits;
}

bool same_point_bits(const Point &a, const Point &b) {
  return same_float_bits(a.x, b.x) && same_float_bits(a.y, b.y) &&
         same_float_bits(a.z, b.z) && same_float_bits(a.strength.x, b.strength.x) &&
         same_float_bits(a.strength.y, b.strength.y);
}

bool same_points(const std::vector<Point> &a, const std::vector<Point> &b) {
  if (a.size() != b.size()) return false;
  for (std::size_t i = 0; i < a.size(); ++i)
    if (!same_point_bits(a[i], b[i])) return false;
  return true;
}

void launch_selected_spread(int kernel, const Point *points,
                            const int *bin_offsets, Grid grid,
                            cuFloatComplex *output) {
  if (kernel == 1) {
    atomic_free_spread::launch_spread_f0(points, bin_offsets, grid, output);
  } else {
    atomic_free_spread::launch_spread(points, bin_offsets, grid, output);
  }
}

}  // namespace

int run_benchmark(int argc, char **argv) {
  int count = argc > 1 ? std::atoi(argv[1]) : 10000;
  int nx = argc > 2 ? std::atoi(argv[2]) : 32;
  int ny = argc > 3 ? std::atoi(argv[3]) : 32;
  int nz = argc > 4 ? std::atoi(argv[4]) : 32;
  int cluster = argc > 5 ? std::atoi(argv[5]) : 0;
  int bin_size = argc > 6 ? std::atoi(argv[6]) : 4;
  int pattern = argc > 7 ? std::atoi(argv[7]) : 0;
  int seed = argc > 8 ? std::atoi(argv[8]) : 42;
  int nrun = argc > 9 ? std::atoi(argv[9]) : 5;
  int kernel = argc > 10 ? std::atoi(argv[10]) : 0;
  int bin_sort = argc > 11 ? std::atoi(argv[11]) : 1;
  if (argc > 12 || count <= 0 || nx < atomic_free_spread::WIDTH ||
      ny < atomic_free_spread::WIDTH || nz < atomic_free_spread::WIDTH ||
      (cluster != 0 && cluster != 1) || bin_size <= 0 ||
      (pattern < 0 || pattern > 3) || nrun <= 0 ||
      (kernel != 0 && kernel != 1) || bin_sort < 0 || bin_sort > 1) {
    std::fprintf(stderr,
                 "Usage: %s [M [N1 [N2 [N3 [cluster [bin_size [pattern [seed "
                 "[nrun [kernel [bin_sort]]]]]]]]]]]\n"
                 "M, bin_size, and nrun must be positive; grid dimensions >= %d; "
                 "cluster is 0 or 1; pattern is 0..3; kernel is 0 or 1; "
                 "bin_sort is 0 (CPU fallback) or 1 (stable radix GPU bin sort).\n",
                 argv[0], atomic_free_spread::WIDTH);
    return 1;
  }
  const Grid grid = atomic_free_spread::make_grid(nx, ny, nz, bin_size);
  const long long num_bins = atomic_free_spread::bin_count(grid);
  const long long num_cells = atomic_free_spread::cell_count(grid);
  // Bin IDs and offsets are 32-bit in this prototype.
  if (num_bins >= 2147483647LL) {
    std::fprintf(stderr, "Too many bins for 32-bit bin IDs.\n");
    return 1;
  }

  std::printf("case   M=%d grid=%dx%dx%d cluster=%d bin_size=%d pattern=%d seed=%d "
              "nrun=%d kernel=%d bins=%lld cells=%lld\n",
              count, nx, ny, nz, cluster, bin_size, pattern, seed, nrun, kernel,
              num_bins, num_cells);

  const auto generate_start = Clock::now();
  HostPoints host = generate_points(count, nx, ny, nz, cluster, pattern, seed);
  const double generate_ms = elapsed_ms(generate_start);

  DeviceBuffer<Point> device_points((size_t)count);
  DeviceBuffer<int> device_bin_ids((size_t)count);
  DeviceBuffer<int> device_offsets((size_t)(num_bins + 1));
  DeviceBuffer<cuFloatComplex> device_output((size_t)num_cells);

  // GPU mode keeps the original points and writes a separate sorted array; the
  // host path reuses device_points for the uploaded sorted sources.
  std::unique_ptr<DeviceBuffer<Point>> device_sorted_sources;
  std::unique_ptr<gpu_bin_sort::BinSorter> bin_sorter;
  if (bin_sort != 0) {
    device_sorted_sources = std::make_unique<DeviceBuffer<Point>>((size_t)count);
    bin_sorter = std::make_unique<gpu_bin_sort::BinSorter>(
        count, (int)num_bins);
  }

  const auto upload_points_start = Clock::now();
  CUDA_CHECK(cudaMemcpy(device_points.get(), host.points.data(), count * sizeof(Point),
                        cudaMemcpyHostToDevice));
  const double upload_points_ms = elapsed_ms(upload_points_start);

  // Warm up the bin kernel and the GPU clocks before the timed launch.
  atomic_free_spread::launch_assign_bins(device_points.get(), count, grid,
                                         device_bin_ids.get());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CudaTimer bin_timer;
  bin_timer.begin();
  atomic_free_spread::launch_assign_bins(device_points.get(), count, grid,
                                         device_bin_ids.get());
  CUDA_CHECK(cudaGetLastError());
  const double assign_bins_ms = bin_timer.end();

  std::vector<int> bin_ids((size_t)count);
  std::vector<int> offsets((size_t)(num_bins + 1), 0);
  std::vector<Point> sorted((size_t)count);
  double copy_bins_ms = 0.0;
  double sort_ms = 0.0;
  double upload_sorted_ms = 0.0;
  double gpu_bin_sort_ms = 0.0;
  bool bins_valid = true;
  bool table_valid = true;
  bool offsets_exact = true;
  bool sorted_exact = true;
  bool binsort_exact_valid = true;

  if (bin_sort != 0) {
    // Warm up CUB's lazy kernel loading and the GPU clocks before timing.
    CUDA_CHECK(bin_sorter->run(
        device_points.get(), device_bin_ids.get(), count, device_offsets.get(),
        device_sorted_sources->get()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // The GPU path writes device_offsets and the sorted sources directly. Time
    // the same number of repetitions as the spread and report the median.
    std::vector<double> prep_sort_ms;
    prep_sort_ms.reserve((size_t)nrun);
    for (int run = 0; run < nrun; ++run) {
      CudaTimer sort_timer;
      sort_timer.begin();
      CUDA_CHECK(bin_sorter->run(
          device_points.get(), device_bin_ids.get(), count, device_offsets.get(),
          device_sorted_sources->get()));
      const double milliseconds = sort_timer.end();
      prep_sort_ms.push_back(milliseconds);
    }
    std::sort(prep_sort_ms.begin(), prep_sort_ms.end());
    gpu_bin_sort_ms = prep_sort_ms[prep_sort_ms.size() / 2];

    // Validate a fresh, untimed run after poisoning the outputs. Stale data from
    // the warmup or a previous timed run cannot mask a missing write.
    CUDA_CHECK(cudaMemset(device_offsets.get(), 0xFF,
                          offsets.size() * sizeof(int)));
    CUDA_CHECK(cudaMemset(device_sorted_sources->get(), 0xFF,
                          (size_t)count * sizeof(Point)));
    CUDA_CHECK(bin_sorter->run(
        device_points.get(), device_bin_ids.get(), count, device_offsets.get(),
        device_sorted_sources->get()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Validation copies and the host reference run outside the timed region.
    CUDA_CHECK(cudaMemcpy(bin_ids.data(), device_bin_ids.get(), count * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(offsets.data(), device_offsets.get(),
                          offsets.size() * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(sorted.data(), device_sorted_sources->get(),
                          count * sizeof(Point), cudaMemcpyDeviceToHost));

    long long invalid_bin_ids = 0;
    for (int bin : bin_ids)
      if (bin < 0 || bin >= num_bins) ++invalid_bin_ids;
    bins_valid = invalid_bin_ids == 0;
    if (!bins_valid) {
      std::fprintf(stderr, "assign_bins produced %lld bin IDs outside [0, %lld).\n",
                   invalid_bin_ids, num_bins);
      return 1;
    }

    // CPU stable-sort reference.
    std::vector<int> reference_offsets((size_t)(num_bins + 1), 0);
    for (int bin : bin_ids) ++reference_offsets[bin + 1];
    std::partial_sum(reference_offsets.begin(), reference_offsets.end(),
                     reference_offsets.begin());
    std::vector<int> reference_cursor = reference_offsets;
    std::vector<Point> reference_sorted((size_t)count);
    for (int i = 0; i < count; ++i) {
      const int position = reference_cursor[bin_ids[i]]++;
      reference_sorted[position] = host.points[i];
    }

    offsets_exact = offsets == reference_offsets;
    sorted_exact = same_points(sorted, reference_sorted);
    binsort_exact_valid = offsets_exact && sorted_exact;
  } else {
    const auto copy_bins_start = Clock::now();
    CUDA_CHECK(cudaMemcpy(bin_ids.data(), device_bin_ids.get(), count * sizeof(int),
                          cudaMemcpyDeviceToHost));
    copy_bins_ms = elapsed_ms(copy_bins_start);

    long long invalid_bin_ids = 0;
    for (int bin : bin_ids)
      if (bin < 0 || bin >= num_bins) ++invalid_bin_ids;
    bins_valid = invalid_bin_ids == 0;
    if (!bins_valid) {
      std::fprintf(stderr, "assign_bins produced %lld bin IDs outside [0, %lld).\n",
                   invalid_bin_ids, num_bins);
      return 1;
    }

    // Counting sort on the host. offsets[b]..offsets[b+1] is bin b's range.
    // Repeat the host sort so the reported preparation time uses the same
    // median-of-nrun treatment as the spread and is not a single scheduling
    // outlier. The final repetition leaves offsets and sorted valid.
    const auto run_host_sort = [&]() {
      std::fill(offsets.begin(), offsets.end(), 0);
      for (int bin : bin_ids) ++offsets[bin + 1];
      std::partial_sum(offsets.begin(), offsets.end(), offsets.begin());
      std::vector<int> cursor = offsets;
      for (int i = 0; i < count; ++i)
        sorted[cursor[bin_ids[i]]++] = host.points[i];
    };
    std::vector<double> prep_sort_ms;
    prep_sort_ms.reserve((size_t)nrun);
    for (int run = 0; run < nrun; ++run) {
      const auto sort_start = Clock::now();
      run_host_sort();
      prep_sort_ms.push_back(elapsed_ms(sort_start));
    }
    std::sort(prep_sort_ms.begin(), prep_sort_ms.end());
    sort_ms = prep_sort_ms[prep_sort_ms.size() / 2];

    const auto upload_sorted_start = Clock::now();
    CUDA_CHECK(cudaMemcpy(device_points.get(), sorted.data(), count * sizeof(Point),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_offsets.get(), offsets.data(),
                          offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
    upload_sorted_ms = elapsed_ms(upload_sorted_start);
  }

  table_valid = offsets.front() == 0 && offsets.back() == count;
  for (size_t i = 1; i < offsets.size(); ++i)
    if (offsets[i] < offsets[i - 1]) table_valid = false;

  const double prepare_ms = upload_points_ms + assign_bins_ms + copy_bins_ms +
                            sort_ms + upload_sorted_ms + gpu_bin_sort_ms;
  const double setup_ms = generate_ms + prepare_ms;
  std::printf("host   generate=%.3f ms\n", generate_ms);
  if (bin_sort != 0) {
    std::printf("prep   upload_points=%.3f ms assign_bins=%.3f ms "
                "gpu_bin_sort=%.3f ms total=%.3f ms\n",
                upload_points_ms, assign_bins_ms, gpu_bin_sort_ms, prepare_ms);
    std::printf("sort   mode=radix\n");
  } else {
    std::printf("prep   upload_points=%.3f ms assign_bins=%.3f ms copy_bins=%.3f ms "
                "sort=%.3f ms upload_sorted=%.3f ms total=%.3f ms\n",
                upload_points_ms, assign_bins_ms, copy_bins_ms, sort_ms,
                upload_sorted_ms, prepare_ms);
    std::printf("sort   mode=cpu\n");
  }
  if (bin_sort == 1) {
    std::printf("binsort exact offsets=%s sorted_sources=%s\n",
                pass_fail(offsets_exact), pass_fail(sorted_exact));
  }
  if (!table_valid || !binsort_exact_valid) {
    std::fprintf(stderr, "preparation validation failed; skipping spread\n");
    return 1;
  }

  const ReferenceResult reference = run_reference(host, nx, ny, nz, num_cells);
  const Point *spread_points =
      bin_sort != 0 ? device_sorted_sources->get() : device_points.get();
  const int *spread_offsets = device_offsets.get();

  // Time warm-ups separately so the reported spread interval is a steady state.
  std::vector<double> spread_ms;
  spread_ms.reserve((size_t)nrun);
  CudaTimer spread_timer;
  for (int run = -2; run < nrun; ++run) {
    spread_timer.begin();
    launch_selected_spread(kernel, spread_points, spread_offsets, grid,
                           device_output.get());
    CUDA_CHECK(cudaGetLastError());
    const double milliseconds = spread_timer.end();
    if (run >= 0) spread_ms.push_back(milliseconds);
  }
  std::sort(spread_ms.begin(), spread_ms.end());
  const double spread_median_ms = spread_ms[spread_ms.size() / 2];
  const double spread_min_ms = spread_ms.front();
  const double spread_max_ms = spread_ms.back();
  std::printf("ref    method=3 setpts=%.3f ms execute=%.3f ms total=%.3f ms\n",
              reference.setpts_ms, reference.execute_ms, reference.total_ms);
  std::printf("spread warmup=2 timed=%d min=%.3f median=%.3f max=%.3f ms "
              "throughput=%.2f Mpoint/s %.2f Mcell/s\n",
              nrun, spread_min_ms, spread_median_ms, spread_max_ms,
              count / spread_median_ms / 1000.0,
              (double)num_cells / spread_median_ms / 1000.0);
  std::printf("timing setup=%.3f ms end_to_end=%.3f ms spread_vs_ref_exec=%.2fx\n",
              setup_ms, setup_ms + spread_median_ms,
              spread_median_ms > 0.0 ? reference.execute_ms / spread_median_ms : 0.0);

  // Correctness run: poison the output first so an unwritten cell is visible.
  CUDA_CHECK(cudaMemset(device_output.get(), 0xFF,
                        (size_t)num_cells * sizeof(cuFloatComplex)));
  launch_selected_spread(kernel, spread_points, spread_offsets, grid,
                         device_output.get());
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<cuFloatComplex> result((size_t)num_cells);
  CUDA_CHECK(cudaMemcpy(result.data(), device_output.get(),
                        (size_t)num_cells * sizeof(cuFloatComplex),
                        cudaMemcpyDeviceToHost));

  long long unwritten_cells = 0;
  long long nonfinite_cells = 0;
  for (const cuFloatComplex &value : result) {
    if (is_unwritten_sentinel(value)) {
      ++unwritten_cells;
    } else if (!is_finite(value)) {
      ++nonfinite_cells;
    }
  }
  const bool coverage_valid = unwritten_cells == 0;
  const bool finite_valid = nonfinite_cells == 0;

  double error_squared = 0.0, reference_squared = 0.0;
  double max_abs_error = 0.0, max_reference = 0.0;
  long long unwritten_reference_cells = 0;
  long long nonfinite_reference_cells = 0;
  for (long long i = 0; i < num_cells; ++i) {
    const cuFloatComplex ref = reference.grid[(size_t)i];
    if (is_unwritten_sentinel(ref)) {
      ++unwritten_reference_cells;
      continue;
    }
    if (!is_finite(ref)) ++nonfinite_reference_cells;
    const double dr = (double)result[(size_t)i].x - (double)ref.x;
    const double di = (double)result[(size_t)i].y - (double)ref.y;
    error_squared += dr * dr + di * di;
    reference_squared += (double)ref.x * ref.x + (double)ref.y * ref.y;
    max_abs_error = std::max(max_abs_error, std::hypot(dr, di));
    max_reference = std::max(max_reference,
                             std::hypot((double)ref.x, (double)ref.y));
  }
  const bool reference_valid =
      unwritten_reference_cells == 0 && nonfinite_reference_cells == 0;
  const double relative_l2 = reference_squared
                                 ? std::sqrt(error_squared / reference_squared)
                                 : std::sqrt(error_squared);
  const double max_relative_error =
      max_reference > 0.0 ? max_abs_error / max_reference : max_abs_error;
  const bool accuracy_valid =
      std::isfinite(relative_l2) && relative_l2 < REL_L2_TOLERANCE;

  // Global consistency with method 3. This is not an absolute mass check:
  // gpu_spreadinterponly returns the raw spread grid, whose total is scaled by
  // the (non-unit) kernel-weight sum. The relative difference is still sensitive
  // to dropped or duplicated points.
  double result_real = 0.0, result_imag = 0.0;
  for (const cuFloatComplex &value : result) {
    if (!is_unwritten_sentinel(value) && is_finite(value)) {
      result_real += value.x;
      result_imag += value.y;
    }
  }
  double reference_real = 0.0, reference_imag = 0.0, reference_abs_sum = 0.0;
  for (const cuFloatComplex &value : reference.grid) {
    if (!is_unwritten_sentinel(value) && is_finite(value)) {
      reference_real += value.x;
      reference_imag += value.y;
      reference_abs_sum += std::hypot((double)value.x, (double)value.y);
    }
  }
  const double total_scale = reference_abs_sum > 0.0 ? reference_abs_sum : 1.0;
  const double total_rel = std::hypot(result_real - reference_real,
                                      result_imag - reference_imag) /
                           total_scale;
  const bool total_valid = total_rel < TOTAL_TOLERANCE;

  std::printf("correct relL2=%.3e max_abs=%.3e max_rel=%.3e total_rel=%.3e "
              "unwritten=%lld nonfinite=%lld reference_unwritten=%lld "
              "reference_nonfinite=%lld\n",
              relative_l2, max_abs_error, max_relative_error, total_rel,
              unwritten_cells, nonfinite_cells, unwritten_reference_cells,
              nonfinite_reference_cells);
  const bool all_valid = bins_valid && table_valid && binsort_exact_valid &&
                         coverage_valid && finite_valid && reference_valid &&
                         total_valid && accuracy_valid;
  std::printf("check  bin_ids=%s table=%s coverage=%s finite=%s reference=%s "
              "total=%s accuracy=%s overall=%s\n",
              pass_fail(bins_valid), pass_fail(table_valid), pass_fail(coverage_valid),
              pass_fail(finite_valid), pass_fail(reference_valid), pass_fail(total_valid),
              pass_fail(accuracy_valid), pass_fail(all_valid));
  return all_valid ? 0 : 1;
}

int main(int argc, char **argv) {
  try {
    return run_benchmark(argc, argv);
  } catch (const std::exception &error) {
    std::fprintf(stderr, "error: %s\n", error.what());
    return 1;
  }
}
