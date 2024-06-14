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


#include <climits>
#include <algorithm>
#include <queue>
#include <sys/time.h>
#include <thread>
#include <atomic>
#include "ECLgraph.h"

const data_type maxval = std::numeric_limits<data_type>::max();

using pair = std::pair<int, int>;

const unsigned char undecided = 0;
const unsigned char included = 1;
const unsigned char excluded = 2;

static double CPPmis_edge(const ECLgraph& g, const int* const sp, data_type* const priority, unsigned char* const status, const int threadCount);

static double median(double array[], const int n)
{
  double median = 0;
  std::sort(array, array + n);
  if (n % 2 == 0) median = (array[(n - 1) / 2] + array[n / 2]) / 2.0;
  else median = array[n / 2];
  return median;
}

int main(int argc, char* argv[])
{
  printf("mis edge-based CPP (%s)\n", __FILE__);
  if (argc != 3 && argc != 4) {fprintf(stderr, "USAGE: %s input_file_name runs thread_count(optional)\n", argv[0]); exit(-1);}

  // process command line
  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);

  int threadCount = std::thread::hardware_concurrency(); //defaults to max threads
  if(argc == 4)
    if(const int countInt = atoi(argv[3])) //checks for valid int
      threadCount = countInt;             //takes optional argument for thread count
  printf("Threads: %d\n", threadCount);

  // create starting point array
  int* const sp = new int [g.edges];
  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      sp[j] = i;
    }
  }

  // allocate memory
  data_type* const priority = new data_type [g.nodes];
  unsigned char* const status = new unsigned char [g.nodes];
  
  const int runs = atoi(argv[2]);
  double runtimes [runs];
  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPPmis_edge(g, sp, priority, status, threadCount);
  }
  const double med = median(runtimes, runs);
  printf("runtime: %.6fs\n", med);
  printf("Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);

  // free memory
  delete [] priority;
  delete [] status;
  delete [] sp;
  freeECLgraph(g);
  return 0;
}

// https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
static unsigned int hash(unsigned int val)
{
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  val = ((val >> 16) ^ val) * 0x45d9f3b;
  return (val >> 16) ^ val;
}

static void updateUndecided(unsigned char* const status, unsigned char* const status_n, const int size, const int threadID, const int threadCount)
{
  const int begNode = threadID * (long)size / threadCount;
  const int endNode = (threadID + 1) * (long)size / threadCount;
  
  for (int i = begNode; i < endNode; ++i)
  {
    if (status[i] == undecided)
      status[i] = status_n[i];
  }
}

static void updateFromWorklist(const ECLgraph& g, const int* const sp, unsigned char* const status, unsigned char* const status_n, const int* const worklist, const int wlsize, const int threadID, const int threadCount)
{
  const int begWork = threadID * (long)wlsize / threadCount;
  const int endWork = (threadID + 1) * (long)wlsize / threadCount;
  
  for (int w = begWork; w < endWork; ++w)
  {
    const int e = worklist[w];
    const int src = sp[e];
    const int dst = g.nlist[e];
    
    status[src] = status_n[src];
    status[dst] = status_n[dst];
  }
}

template <typename T>
static inline T atomicMin(T* addr, T val)
{
  T oldv = ((std::atomic<T>*)addr)->load();
  while (oldv > val && !(((std::atomic<T>*)addr)->compare_exchange_weak(oldv, val))) {}
  return oldv;
}

template <typename T>
static inline T atomicMax(T* addr, T val)
{
  T oldv = ((std::atomic<T>*)addr)->load();
  while (oldv < val && !(((std::atomic<T>*)addr)->compare_exchange_weak(oldv, val))) {}
  return oldv;
}

template <typename T>
static inline T atomicMin(std::atomic<T>* addr, T val)
{
  T oldv = addr->load();
  while (oldv > val && !(addr->compare_exchange_weak(oldv, val))) {}
  return oldv;
}

template <typename T>
static inline T atomicMax(std::atomic<T>* addr, T val)
{
  T oldv = addr->load();
  while (oldv < val && !(addr->compare_exchange_weak(oldv, val))) {}
  return oldv;
}
