/*
This file is part of the Indigo3 benchmark suite version 1.0.

BSD 3-Clause License

Copyright (c) 2024, Yiqian Liu, Noushin Azami, Avery Vanausdal, and Martin Burtscher.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of the Indigo3 benchmark suite is available at https://github.com/burtscher/Indigo3Suite/.

Publication: This work is described in detail in the following paper.
Yiqian Liu, Noushin Azami, Avery Vanausdal, and Martin Burtscher. "Indigo3: A Parallel Graph Analytics Benchmark Suite for Exploring Implementation Styles and Common Bugs." ACM Transactions on Parallel Computing. May 2024.
*/


#ifndef ATOMIC
#define ATOMIC

#if defined(__HIP_PLATFORM_AMD__)
  #define COMPILETIME_warpSize warpSize
#else
  #define COMPILETIME_warpSize 32
#endif //defined(__HIP_PLATFORM_AMD__)

#include <cuda/atomic>
__device__ inline basic_t atomicRead(basic_t* const addr)
{
  return ((cuda::atomic<basic_t>*)addr)->load(cuda::memory_order_relaxed);
}

__device__ inline void atomicWrite(basic_t* const addr, const basic_t val)
{
  ((cuda::atomic<basic_t>*)addr)->store(val, cuda::memory_order_relaxed);
}

#if defined(__HIP_PLATFORM_AMD__)

  // AMD doesn't need nor define __syncwarp
  static inline __device__ void __syncwarp() {}

  // *_sync versions are not fully supported in HIP and unnecessary on AMD GPUs https://github.com/ROCm/hip/issues/1491#issuecomment-778652063
  #define __shfl_sync(mask, val, src) __shfl(val, src)
  #define __shfl_up_sync(mask, val, src) __shfl_up(val, src)
  #define __shfl_down_sync(mask, val, src) __shfl_down(val, src)
  #define __shfl_xor_sync(mask, val, src) __shfl_xor(val, src)
  #define __any_sync(mask, predicate) __any(predicate)
  #define __all_sync(mask, predicate) __all(predicate)

  // HIP does block scope on shared memory
  #define atomicAdd_block atomicAdd

#endif //defined(__HIP_PLATFORM_AMD__)

#endif // ATOMIC
