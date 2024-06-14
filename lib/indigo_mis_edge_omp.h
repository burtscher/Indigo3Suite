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
#include <stdbool.h>
#include "ECLgraph.h"
#include "csort.h"
#include "omp_helper.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define SWAP(a, b) do { typeof(a) temp = a; a = b; b = temp; } while (0)
const data_type maxval = INT_MAX;

const data_type undecided = 0;
const data_type included = 1;
const data_type excluded = 2;

static double OMPmis_edge(const ECLgraph g, const int* const sp, data_type* const priority, data_type* const status);

int main(int argc, char* argv[])
{
  printf("mis edge-based OMP (%s)\n", __FILE__);
  if (argc < 3) {fprintf(stderr, "USAGE: %s input_file_name runs\n", argv[0]); exit(-1);}

  // process command line
  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);

  // create starting point array
  int* const sp = (int*)malloc(g.edges * sizeof(int));
  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      sp[j] = i;
    }
  }

  // allocate memory
  data_type* const priority = (data_type*)malloc(g.nodes * sizeof(data_type));
  data_type* const status = (data_type*)malloc(g.nodes * sizeof(data_type));

  const int runs = atoi(argv[2]);
  double runtimes [runs];
  for (int i = 0; i < runs; i++) {
    runtimes[i] = OMPmis_edge(g, sp, priority, status);
  }
  const double med = median(runtimes, runs);
  printf("runtime: %.6fs\n", med);
  printf("Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);

  // free memory
  free(priority);
  free(status);
  free(sp);
  freeECLgraph(&g);
  return 0;
}

// https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
static unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

static void updateUndecided(data_type* const status, data_type* const status_n, const int size)
{
  #pragma omp parallel for
  for (int i = 0; i < size; ++i)
  {
    if (status[i] == undecided)
      status[i] = status_n[i];
  }
}

static void updateFromWorklist(const ECLgraph g, const int* const sp, data_type* const status, data_type* const status_n, const int* const worklist, const int wlsize)
{
  #pragma omp parallel for
  for (int i = 0; i < wlsize; ++i)
  {
    const int e = worklist[i];
    const int src = sp[e];
    const int dst = g.nlist[e];

    atomicWrite(&status[src], status_n[src]);
    atomicWrite(&status[dst], status_n[dst]);
  }
}
