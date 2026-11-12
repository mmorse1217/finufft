#ifndef GPU_BIN_SORT_CUH
#define GPU_BIN_SORT_CUH

#include "atomic_free_spread.cuh"
#include <memory>

namespace gpu_bin_sort {

// Stable GPU bin sort on the default stream. Reuse one workspace sequentially.
// Inputs: count <= capacity, bin IDs in [0, bins), disjoint input/output arrays.
// Outputs: bins+1 offsets and count sorted Points, preserving order within bins.
// run() is asynchronous; callers check its result and synchronize before host reads.
// Construction allocates scratch and throws on failure. No allocation in run().
class BinSorter {
 public:
  BinSorter(int capacity, int bins);
  cudaError_t run(const atomic_free_spread::Point *points, const int *bin_ids,
                  int count, int *offsets, atomic_free_spread::Point *sorted);

 private:
  struct Free { void operator()(void *p) const { cudaFree(p); } };
  std::unique_ptr<void, Free> storage_, temp_;
  int capacity_, bins_, bits_;
  int *keys_, *keys_out_, *indices_, *indices_out_, *counts_;
  size_t radix_bytes_ = 0, scan_bytes_ = 0;
};

}  // namespace gpu_bin_sort
#endif
