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


#include <limits.h>
#include <sys/time.h>
#include <cuda.h>
#include "ECLgraph.h"
#include "cuda_atomic.h"
#include "csort.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define SWAP(a, b) do { __typeof__(a) temp = a; a = b; b = temp; } while (0)

static const data_type EPSILON = 0.0001;
static const data_type kDamp = 0.85;
static const int MAX_ITER = 100;
static const int WarpSize = 32;

double PR_GPU(const ECLgraph g, data_type *scores, int* degree);

static int GPUinfo(const int d)
{
  cudaSetDevice(d);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, d);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {printf("ERROR: there is no CUDA capable device\n\n");  exit(-1);}
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int SMs = deviceProp.multiProcessorCount;
  printf("GPU: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n", deviceProp.name, SMs, mTpSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);
  return SMs * mTpSM;
}

static __device__ inline data_type block_sum_reduction(data_type val, void* buffer)  // returns sum to all threads
{
  const int lane = threadIdx.x % WarpSize;
  const int warp = threadIdx.x / WarpSize;
  const int warps = ThreadsPerBlock / WarpSize;
  data_type* const s_carry = (data_type*)buffer;

  val += __shfl_xor_sync(~0, val, 1);  // MB: use reduction on 8.6 CC
  val += __shfl_xor_sync(~0, val, 2);
  val += __shfl_xor_sync(~0, val, 4);
  val += __shfl_xor_sync(~0, val, 8);
  val += __shfl_xor_sync(~0, val, 16);
  if (lane == 0) s_carry[warp] = val;
  __syncthreads();  // s_carry written

  if (warps > 1) {
    if (warp == 0) {
      val = (lane < warps) ? s_carry[lane] : 0;
      val += __shfl_xor_sync(~0, val, 1);  // MB: use reduction on 8.6 CC
      val += __shfl_xor_sync(~0, val, 2);
      val += __shfl_xor_sync(~0, val, 4);
      val += __shfl_xor_sync(~0, val, 8);
      val += __shfl_xor_sync(~0, val, 16);
      s_carry[lane] = val;
    }
    __syncthreads();  // s_carry updated
  }

  return s_carry[0];
}

static void CheckCuda()
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d: %s\n", e, cudaGetErrorString(e));
    exit(-1);
  }
}

int main(int argc, char *argv[]) {
  printf("PageRank CUDA v0.1 (%s)\n", __FILE__);
  printf("Copyright 2022 Texas State University\n\n");

  if (argc < 3) {printf("USAGE: %s input_graph runs\n\n", argv[0]);  exit(-1);}

  // read input
  ECLgraph g = readECLgraph(argv[1]);
  printf("graph: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  printf("avg degree: %.2f\n", 1.0 * g.edges / g.nodes);

  // count degree
  int* degree = (int*)malloc(g.nodes * sizeof(int));
  for (int i = 0; i < g.nodes; i++) {
    degree[i] = g.nindex[i + 1] - g.nindex[i];
  }

  const int runs = atoi(argv[2]);
  
  // init scores
  const data_type init_score = 1.0f / (data_type)g.nodes;
  data_type* scores = (data_type*)malloc(g.nodes * sizeof(data_type));
  double runtimes [runs];
  
  for (int i = 0; i < runs; i++) {
    // init scores
    for (int j = 0; j < g.nodes; j++) scores[j] = init_score;
    
    runtimes[i] = PR_GPU(g, scores, degree);
  }
  
  const double med = median(runtimes, runs);
  printf("runtime: %.6fs\n\n", med);
  printf("Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);

  // compare and verify
  const data_type base_score = (1.0f - kDamp) / (data_type)g.nodes;
  data_type* incoming_sums = (data_type*)malloc(g.nodes * sizeof(data_type));
  for(int i = 0; i < g.nodes; i++) incoming_sums[i] = 0.0;
  double error = 0.0;

  for (int src = 0; src < g.nodes; src++) {
    data_type outgoing_contrib = scores[src] / degree[src];
    for (int i = g.nindex[src]; i < g.nindex[src + 1]; i++) {
      incoming_sums[g.nlist[i]] += outgoing_contrib;
    }
  }

  for (int i = 0; i < g.nodes; i++) {
    data_type new_score = base_score + kDamp * incoming_sums[i];
    error += fabs(new_score - scores[i]);
    incoming_sums[i] = 0;
  }
  if (error < EPSILON) printf("All good.\n");
  else printf("Total Error: %f\n", error);
  
  // free memory
  free(degree);
  free(scores);
  free(incoming_sums);
  freeECLgraph(&g);
  return 0;
}
