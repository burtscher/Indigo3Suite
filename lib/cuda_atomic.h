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

// Define a macro to switch between implementations
#ifdef SLOWER_ATOMIC
// Implementation using slower atomic add operation
__device__ inline basic_t atomicRead(basic_t* const addr)
{
  return atomicAdd(addr, 0);
}

__device__ inline void atomicWrite(basic_t* const addr, const basic_t val)
{
  atomicExch(addr, val);
}

static __device__ inline basic_t atomicCAS_CUDA(cuda::atomic<basic_t>* addr, basic_t compare, basic_t val)
{
  atomicCAS((basic_t*)addr, compare, val);
  return compare;
}

#else
// Default implementation using faster atomic add operation
#include <cuda/atomic>
__device__ inline basic_t atomicRead(basic_t* const addr)
{
  return ((cuda::atomic<basic_t>*)addr)->load(cuda::memory_order_relaxed);
}

__device__ inline void atomicWrite(basic_t* const addr, const basic_t val)
{
  ((cuda::atomic<basic_t>*)addr)->store(val, cuda::memory_order_relaxed);
}

static __device__ inline basic_t atomicCAS_CUDA(cuda::atomic<basic_t>* addr, basic_t compare, basic_t val)
{
  (addr)->compare_exchange_strong(compare, val);
  return compare;
}

#endif // SLOWER_ATOMIC

#endif // ATOMIC
