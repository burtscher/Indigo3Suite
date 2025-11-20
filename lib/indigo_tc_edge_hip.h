#include "hip/hip_runtime.h"
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
#include <hip/hip_runtime.h>
#include "ECLgraph.h"
#include "hip_atomic.h"
#include "csort.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define SWAP(a, b) do { __typeof__(a) temp = a; a = b; b = temp; } while (0)

static double GPUtc_edge(basic_t* count, const int edges, const int* const nindex, const int* const nlist, const int* const sp);

static int GPUinfo(const int d, bool print = false)
{
  hipSetDevice(d);
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, d);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {printf("ERROR: there is no CUDA capable device\n\n");  exit(-1);}
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int SMs = deviceProp.multiProcessorCount;
  if (print) printf("GPU: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n", deviceProp.name, SMs, mTpSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);
  return SMs * mTpSM;
}

static inline __device__ bool d_find(const int target, const int beg, const int end, const int* const __restrict__ nlist)
{
  int left = beg;
  int right = end;
  while (left <= right) {
    int middle = (left + right) / 2;
    if (nlist[middle] == target) return true;
    if (nlist[middle] < target) left = middle + 1;
    else right = middle - 1;
  }
  return false;
}

#ifndef NO_VERIFY
static inline int h_common(const int beg1, const int end1, const int beg2, const int end2, const int* const __restrict__ nlist)
{
  int common = 0;
  int pos1 = beg1;
  int pos2 = beg2;
  while ((pos1 < end1) && (pos2 < end2)) {
    while ((pos1 < end1) && (nlist[pos1] < nlist[pos2])) pos1++;
    if (pos1 < end1) {
      while ((pos2 < end2) && (nlist[pos2] < nlist[pos1])) pos2++;
      if ((pos2 < end2) && (nlist[pos1] == nlist[pos2])) {
        pos1++;
        pos2++;
        common++;
      } else {
        pos1++;
      }
    }
  }
  return common;
}

static basic_t h_triCounting(const int nodes, const int* const __restrict__ nindex, const int* const __restrict__ nlist)
{
  basic_t count = 0;

  for (int v = 0; v < nodes; v++) {
    const int beg1 = nindex[v];
    const int end1 = nindex[v + 1];
    int start1 = end1;
    while ((beg1 < start1) && (v < nlist[start1 - 1])) start1--;
    for (int j = start1; j < end1; j++) {
      const int u = nlist[j];
      const int beg2 = nindex[u];
      const int end2 = nindex[u + 1];
      int start2 = end2;
      while ((beg2 < start2) && (u < nlist[start2 - 1])) start2--;
      count += h_common(j + 1, end1, start2, end2, nlist);
    }
  }
  return count;
}
#endif // NO_VERIFY undefined

int main(int argc, char* argv [])
{
  printf("Triangle counting edge-centric (%s)\n", __FILE__);

  if (argc < 3) {printf("USAGE: %s input_graph runs\n\n", argv[0]);  exit(-1);}

  // read input
  ECLgraph g = readECLgraph(argv[1]);
  printf("graph: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  printf("avg degree: %.2f\n", 1.0 * g.edges / g.nodes);

  // info only
  int mdeg = 0;
  for (int v = 0; v < g.nodes; v++) {
    mdeg = MAX(mdeg, g.nindex[v + 1] - g.nindex[v]);
  }
  printf("max degree: %d\n\n", mdeg);

  // check if sorted
  for (int v = 0; v < g.nodes; v++) {
    for (int i = g.nindex[v] + 1; i < g.nindex[v + 1]; i++) {
      if (g.nlist[i - 1] >= g.nlist[i]) {
        printf("ERROR: adjacency list not sorted or contains self edge\n");
        exit(-1);
      }
    }
  }

  // create starting point array
  int* const sp = (int*)malloc(g.edges * sizeof(int));
  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      sp[j] = i;
    }
  }

  // allocate memory
  basic_t* count = (basic_t*)malloc(sizeof(basic_t));
  *count = 0;
  int* d_sp;
  ECLgraph d_g = g;
  if (hipSuccess != hipMalloc((void **)&d_g.nindex, (g.nodes + 1) * sizeof(int))) fprintf(stderr, "ERROR: could not allocate nindex\n");
  if (hipSuccess != hipMalloc((void **)&d_g.nlist, g.edges * sizeof(int))) fprintf(stderr, "ERROR: could not allocate nlist\n");
  if (hipSuccess != hipMalloc((void **)&d_sp, sizeof(int) * g.edges)) {fprintf(stderr, "ERROR: could not allocate d_sp\n"); exit(-1);}

  if (hipSuccess != hipMemcpy(d_g.nindex, g.nindex, (g.nodes + 1) * sizeof(int), hipMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of index to device failed\n");
  if (hipSuccess != hipMemcpy(d_g.nlist, g.nlist, g.edges * sizeof(int), hipMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of nlist to device failed\n");
  hipMemcpy(d_sp, sp, sizeof(int) * g.edges, hipMemcpyHostToDevice);

  // launch kernel
  const int runs = atoi(argv[2]);
  printf("runs: %d\n", runs);
  double runtimes [runs];

  for (int i = 0; i < runs; i++) {
  	runtimes[i] = GPUtc_edge(count, g.edges, d_g.nindex, d_g.nlist, d_sp);
  }

  const double med = median(runtimes, runs);
  printf("GPU runtime: %.6f s\n", med);
  printf("GPU Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);

  printf("the pattern occurs %lld times\n\n", (long long)*count);

  #ifndef NO_VERIFY
  basic_t h_count = h_triCounting(g.nodes, g.nindex, g.nlist);
  if (h_count != *count) printf("ERROR: host %lld device %lld", (long long)h_count, (long long)*count);
  else printf("verification passed\n\n");
  #endif // NO_VERIFY undefined

  // clean up
  free(sp);
  freeECLgraph(&g);
  hipFree(d_g.nindex);
  hipFree(d_g.nlist);
  hipFree(d_sp);
  return 0;
}
