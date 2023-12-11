#ifndef ATOMIC
#define ATOMIC

// Define a macro to switch between implementations
#ifdef SLOWER_ATOMIC
// Implementation using slower atomic add operation
__device__ inline data_type atomicRead(data_type* const addr)
{
  return atomicAdd(addr, 0);
}

__device__ inline void atomicWrite(data_type* const addr, const data_type val)
{
  atomicExch(addr, val);
}

static __device__ inline int atomicCAS_CUDA(int* addr, int compare, int val)
{
  atomicCAS(addr, compare, val);
  return compare;
}

static __global__ void fill_darray(data_type* arr, const data_type val, const size_t size)
{
  int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < size) {
    arr[idx] = val;
  }
}

#else
// Default implementation using faster atomic add operation
#include <cuda/atomic>
__device__ inline data_type atomicRead(data_type* const addr)
{
  return ((cuda::atomic<data_type>*)addr)->load(cuda::memory_order_relaxed);
}

__device__ inline void atomicWrite(data_type* const addr, const data_type val)
{
  ((cuda::atomic<data_type>*)addr)->store(val, cuda::memory_order_relaxed);
}

static __device__ inline data_type atomicCAS_CUDA(cuda::atomic<data_type>* addr, data_type compare, data_type val)
{
  (addr)->compare_exchange_strong(compare, val);
  return compare;
}

static __global__ void fill_darray(cuda::atomic<data_type>* arr, const data_type val, const size_t size)
{
  int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < size) {
    arr[idx] = val;
  }
}

#endif // SLOWER_ATOMIC

#endif // ATOMIC
