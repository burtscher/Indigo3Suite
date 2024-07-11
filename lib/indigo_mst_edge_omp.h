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


#include <sys/time.h>
#include <stdbool.h>
#include <limits.h>
#include "ECLgraph.h"
#include "csort.h"
#include "omp_helper.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define SWAP(a, b) do { __typeof__(a) temp = a; a = b; b = temp; } while (0)

const int maxval = INT_MAX;

static double CPUmst(const ECLgraph g, const int* const sp, bool* const included);

int main(int argc, char* argv[])
{
  printf("MST edge-based OMP (%s)\n", __FILE__);
  if (argc < 3) {fprintf(stderr, "USAGE: %s input_file_name runs thread_count(optional)\n", argv[0]); exit(-1);}

  // process command line
  ECLgraph g = readECLgraph(argv[1]);
  if (g.eweight == NULL) {
    printf("Generating weights.\n");
    g.eweight = malloc(sizeof(int) * g.edges);
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
  const int runs = atoi(argv[2]);
  
  // create starting point array
  int* const sp = (int*)malloc(g.edges * sizeof(int));
  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      sp[j] = i;
    }
  }

  // allocate memory
  bool* const included = (bool*)malloc(g.edges * sizeof(bool));

  // launch kernel
  double runtimes [runs];
  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPUmst(g, sp, included);
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

  // free memory
  free(sp);
  free(included);
  freeECLgraph(&g);
  return 0;
}

static void updateFromWorklist(unsigned char* const status, unsigned char* const status_n, const int* const worklist, const int wlsize)
{
  #pragma omp parallel for
  for (int i = 0; i < wlsize; ++i)
  {
    int v = worklist[i];
    status[v] = status_n[v];
  }
}

static inline int critical_CAS(int* addr, int compare, int val)
{
  int oldv;
  #pragma omp critical
  {
    oldv = *addr;
    if (oldv == compare) {
      *addr = val;
    }
  }
  return oldv;
}
