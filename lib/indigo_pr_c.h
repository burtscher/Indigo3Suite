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
#include <math.h>
#include <threads.h>
#include <stdatomic.h>
#include "ECLgraph.h"
#include "csort.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define SWAP(a, b) do { __typeof__(a) temp = a; a = b; b = temp; } while (0)

static const score_type EPSILON = 0.0001;
static const score_type kDamp = 0.85;
static const int MAX_ITER = 100;

static double PR_CPU(const ECLgraph g, score_type* scores, int* degree, const int threadCount);

static inline double atomicAddDouble(_Atomic double* addr, double val)
{
  double old = atomic_load(addr);
  while (!(atomic_compare_exchange_weak(addr, &old, old + val))) {}
  return old;
}

static inline score_type atomicAdd(_Atomic score_type* addr, score_type val)
{
  score_type old = atomic_load(addr);
  while (!(atomic_compare_exchange_weak(addr, &old, old + val))) {}
  return old;
}

int main(int argc, char *argv[]) {
  printf("PageRank C v0.1 (%s)\n", __FILE__);
  printf("Copyright 2022 Texas State University\n\n");

  if (argc < 4) {printf("USAGE: %s input_graph runs thread_count\n\n", argv[0]);  exit(-1);}

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
  const int threadCount = atoi(argv[3]);
  printf("Threads: %d\n\n", threadCount);

  const score_type init_score = 1.0f / (score_type)g.nodes;
  score_type* scores = (score_type*)malloc(g.nodes * sizeof(score_type));
  double runtimes [runs];
  
  for (int i = 0; i < runs; i++) {
    // init scores
    for (int j = 0; j < g.nodes; j++) scores[j] = init_score;
    
    runtimes[i] = PR_CPU(g, scores, degree, threadCount);
  }
  
  const double med = median(runtimes, runs);
  printf("runtime: %.6fs\n\n", med);
  printf("Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);
  
  // compare and verify
  const score_type base_score = (1.0f - kDamp) / (score_type)g.nodes;
  score_type* incoming_sums = (score_type*)malloc(g.nodes * sizeof(score_type));
  for(int i = 0; i < g.nodes; i++) incoming_sums[i] = 0;
  double error = 0;
  
  for (int src = 0; src < g.nodes; src++) {
    score_type outgoing = scores[src] / degree[src];
    for (int i = g.nindex[src]; i < g.nindex[src + 1]; i++) {
      incoming_sums[g.nlist[i]] += outgoing;
    }
  }
  
  for (int i = 0; i < g.nodes; i++) {
    score_type new_score = base_score + kDamp * incoming_sums[i];
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
