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
#include "ECLgraph.h"
#include "csort.h"
#include "omp_helper.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

static double CPUtc_vertex(data_type* count, const int nodes, const int* const nindex, const int* const nlist);

static inline int common(const int beg1, const int end1, const int beg2, const int end2, const int* const nlist)
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

int main(int argc, char* argv [])
{
  printf("Triangle counting vertex-centric (%s)\n", __FILE__);

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

  // allocate memory
  data_type* count = malloc(sizeof(data_type));
  *count = 0;

  // launch kernel
  const int runs = atoi(argv[2]);
  double runtimes [runs];

  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPUtc_vertex(count, g.nodes, g.nindex, g.nlist);
  }

  const double med = median(runtimes, runs);
  printf("OMP runtime: %.6f s\n", med);
  printf("OMP Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);

  printf("the pattern occurs %lld times\n\n", (long long)count);
  
  // clean up
  freeECLgraph(&g);
  return 0;
}
