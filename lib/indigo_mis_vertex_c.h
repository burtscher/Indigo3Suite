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
#include <threads.h>
#include <stdatomic.h>
#include <stdbool.h>
#include "ECLgraph.h"
#include "csort.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define SWAP(a, b) do { __typeof__(a) temp = a; a = b; b = temp; } while (0)

const unsigned char undecided = 0;
const unsigned char included = 1;
const unsigned char excluded = 2;

static double CPPmis_vertex(const ECLgraph g, data_type* const priority, unsigned char* const status, const int threadCount);

int main(int argc, char* argv[])
{
  printf("mis vertex-based C (%s)\n", __FILE__);
  if (argc != 4) {fprintf(stderr, "USAGE: %s input_file_name runs thread_count\n", argv[0]); exit(-1);}

  // process command line
  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  
  const int threadCount = atoi(argv[3]);
  printf("Threads: %d\n", threadCount);

  // allocate memory
  data_type* const priority = (data_type*)malloc(g.nodes * sizeof(data_type));
  unsigned char* const status = (unsigned char*)malloc(g.nodes * sizeof(unsigned char));

  // launch kernel
  const int runs = atoi(argv[2]);
  double runtimes [runs];
  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPPmis_vertex(g, priority, status, threadCount);
  }
  const double med = median(runtimes, runs);
  printf("runtime: %.6fs\n", med);
  printf("Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);

  // free memory
  free(priority);
  free(status);
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

static inline int atomicMax(_Atomic int* addr, int val)
{
  int oldv = atomic_load(addr);
  while (oldv < val && !(atomic_compare_exchange_weak(addr, &oldv, val))) {}
  return oldv;
}
