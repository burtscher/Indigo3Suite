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


#include <algorithm>
#include <vector>
#include <numeric>
#include <tuple>
#include <limits.h>
#include <sys/time.h>
#include <hip/hip_runtime.h>
#include "ECLgraph.h"
#include "hip_atomic.h"
#include "csort.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define SWAP(a, b) do { __typeof__(a) temp = a; a = b; b = temp; } while (0)

const data_type maxval = INT_MAX;

static double GPUmst(const ECLgraph g, const int* const sp, bool* const included);

static int GPUinfo(const int d, const bool print = true)
{
  hipSetDevice(d);
  hipDeviceProp_t deviceProp;
  hipGetDeviceProperties(&deviceProp, d);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {printf("ERROR: there is no CUDA capable device\n\n");  exit(-1);}
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int SMs = deviceProp.multiProcessorCount;
  if (print) {
    printf("GPU: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n", deviceProp.name, SMs, mTpSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);
  }
  return SMs * mTpSM;
}

static void CheckCuda()
{
  hipError_t e;
  hipDeviceSynchronize();
  if (hipSuccess != (e = hipGetLastError())) {
    fprintf(stderr, "CUDA error %d: %s\n", e, hipGetErrorString(e));
    exit(-1);
  }
}

static __global__ void fill_darray(idx_type* arr, const basic_t val, const size_t size)
{
  int idx = threadIdx.x + blockIdx.x * ThreadsPerBlock;
  if (idx < size) {
    arr[idx] = val;
  }
}

#ifndef NO_VERIFY
static inline int serial_find(const int idx, int* const parent)
{
  int curr = parent[idx];
  if (curr != idx) {
    int next, prev = idx;
    while (curr != (next = parent[curr])) {
      parent[prev] = next;
      prev = curr;
      curr = next;
    }
  }
  return curr;
}

static inline void serial_join(const int a, const int b, int* const parent)
{
  const int arep = serial_find(a, parent);
  const int brep = serial_find(b, parent);
  if (arep > brep) {  // improves locality
    parent[brep] = arep;
  } else {
    parent[arep] = brep;
  }
}

static void CPUserialMST(const ECLgraph& g, bool* const inMST)
{
  int* const parent = new int [g.nodes];

  timeval start, end;
  gettimeofday(&start, NULL);

  std::fill(inMST, inMST + g.edges, false);
  for (int i = 0; i < g.nodes; i++) parent[i] = i;

  std::vector<std::tuple<int, int, int, int>> list;  // <weight, edge index, from node, to node>
  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      const int n = g.nlist[j];
      if (n > i) {  // only one direction
        list.push_back(std::make_tuple(g.eweight[j], j, i, n));
      }
    }
  }
  std::sort(list.begin(), list.end());

  int count = g.nodes - 1;
  for (int pos = 0; pos < list.size(); pos++) {
    const int a = std::get<2>(list[pos]);
    const int b = std::get<3>(list[pos]);
    const int arep = serial_find(a, parent);
    const int brep = serial_find(b, parent);
    if (arep != brep) {
      const int j = std::get<1>(list[pos]);
      inMST[j] = true;
      serial_join(arep, brep, parent);
      count--;
      if (count == 0) break;
    }
  }

  gettimeofday(&end, NULL);
  double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
  //printf("Serial time: %12.9f s\n", runtime);

  delete [] parent;
}
#endif // NO_VERIFY undefined

int main(int argc, char* argv[])
{
  printf("MST edge-based CUDA (%s)\n", __FILE__);
  if (argc < 3) {fprintf(stderr, "USAGE: %s input_file_name runs\n", argv[0]); exit(-1);}

  // process command line
  ECLgraph g = readECLgraph(argv[1]);
  if (g.eweight == NULL) {
    printf("Generating weights.\n");
    g.eweight = (int*)malloc(g.edges * sizeof(int));
    for (int i = 0; i < g.nodes; i++) {
      for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
        const int nei = g.nlist[j];
        g.eweight[j] = 1 + ((i * nei) % g.nodes);
          if (g.eweight[j] < 0) g.eweight[j] = -g.eweight[j];
      }
    }
  }
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  // const int runveri = atoi(argv[2]);
  // if ((runveri != 0) && (runveri != 1)) {
    // printf("has to be 0 (turn off) or 1 (turn on) verification");
  // }
  
  // create starting point array
  int* const sp = (int*)malloc(g.edges * sizeof(int));
  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      sp[j] = i;
    }
  }

  // allocate memory
  bool* const included = (bool*)malloc(g.edges * sizeof(bool));
  ECLgraph d_g = g;
  if (hipSuccess != hipMalloc((void **)&d_g.nindex, (g.nodes + 1) * sizeof(int))) fprintf(stderr, "ERROR: could not allocate nindex\n");
  if (hipSuccess != hipMalloc((void **)&d_g.nlist, g.edges * sizeof(int))) fprintf(stderr, "ERROR: could not allocate nlist\n");
  if (hipSuccess != hipMalloc((void **)&d_g.eweight, g.edges * sizeof(int))) fprintf(stderr, "ERROR: could not allocate eweight\n");
  if (hipSuccess != hipMemcpy(d_g.nindex, g.nindex, (g.nodes + 1) * sizeof(int), hipMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of nindex to device failed\n");
  if (hipSuccess != hipMemcpy(d_g.nlist, g.nlist, g.edges * sizeof(int), hipMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of nlist to device failed\n");
  if (hipSuccess != hipMemcpy(d_g.eweight, g.eweight, g.edges * sizeof(int), hipMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of eweight to device failed\n");
  int* d_sp;
  if (hipSuccess != hipMalloc((void **)&d_sp, sizeof(int) * g.edges)) {fprintf(stderr, "ERROR: could not allocate d_sp\n"); exit(-1);}
  if (hipSuccess != hipMemcpy(d_sp, sp, sizeof(int) * g.edges, hipMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of sp to device failed\n");
  
  // launch kernel
  const int runs = atoi(argv[2]);
  double runtimes [runs];
  for (int i = 0; i < runs; i++) {
    runtimes[i] = GPUmst(d_g, d_sp, included);
    CheckCuda();
  }
  const double med = median(runtimes, runs);
  printf("runtime: %.6fs\n", med);
  printf("Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);

  // print result
  int count = 0;
  int weight = 0;
  for (int e = 0; e < g.edges; e++) {
    if (included[e]) {
      count++;
      weight += g.eweight[e];
    }
  }
  printf("MSF includes %d edges with %d weight\n", count, weight);
  
  #ifndef NO_VERIFY
  bool* const verify = new bool [g.edges];
  CPUserialMST(g, verify);
  
  int vcount = 0;
  int vweight = 0;
  for (int e = 0; e < g.edges; e++) {
    if (verify[e]) {
      vcount++;
      vweight += g.eweight[e];
    }
  }
  if (vcount != count || vweight != weight) printf("verification solution includes %d edges with %d weight\n", vcount, vweight);
  
  for (int e = 0; e < g.edges; e++) {
    if (included[e] != verify[e]) {fprintf(stderr, "ERROR: verification failed for edge %d: %s instead of %s\n", e, included[e] ? "in" : "out", verify[e] ? "in" : "out"); exit(-1);}
  }
  printf("verification passed\n\n");
  delete [] verify;
  #endif // NO_VERIFY undefined
  
  // free memory
  free(sp);
  free(included);
  hipFree(d_sp);
  hipFree(d_g.nindex);
  hipFree(d_g.nlist);
  hipFree(d_g.eweight);
  freeECLgraph(&g);
  return 0;
}
