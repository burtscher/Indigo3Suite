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
#include <limits.h>
#include <threads.h>
#include <stdatomic.h>
#include <stdbool.h>
#include "ECLgraph.h"
#include "csort.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define SWAP(a, b) do { __typeof__(a) temp = a; a = b; b = temp; } while (0)

static double CPUcc_vertex(const ECLgraph g, data_type* label, const int threadCount);

int main(int argc, char* argv[])
{
  printf("cc vertex-based C (%s)\n", __FILE__);
  if (argc != 4) {fprintf(stderr, "USAGE: %s input_file_name runs thread_count\n", argv[0]); exit(-1);}

  // process command line
  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  
  const int threadCount = atoi(argv[3]);
  printf("Threads: %d\n", threadCount);

  // allocate memory
  data_type* const label = (data_type*)malloc(g.nodes * sizeof(data_type));

  // cc
  const int runs = atoi(argv[2]);
  double runtimes [runs];
  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPUcc_vertex(g, label, threadCount);
  }
  const double med = median(runtimes, runs);
  printf("C runtime: %.6fs\n", med);
  printf("C Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);

  /*
  // print result
  std::set<int> s1;
  for (int v = 0; v < g.nodes; v++) {
    s1.insert(label[v]);
  }
  printf("number of connected components: %d\n", s1.size());
  */
  
  // free memory
  free(label);
  freeECLgraph(&g);
  return 0;
}

static inline data_type atomicMin(_Atomic data_type* addr, data_type val)
{
  data_type oldv = atomic_load(addr);
  while (oldv > val && !(atomic_compare_exchange_weak(addr, &oldv, val))) {}
  return oldv;
}

static inline int atomicMax(_Atomic int* addr, int val)
{
  int oldv = atomic_load(addr);
  while (oldv < val && !(atomic_compare_exchange_weak(addr, &oldv, val))) {}
  return oldv;
}
