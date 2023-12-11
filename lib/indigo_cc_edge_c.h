#include <sys/time.h>
#include <limits.h>
#include <threads.h>
#include <stdatomic.h>
#include <stdbool.h>
#include "ECLgraph.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define SWAP(a, b) do { __typeof__(a) temp = a; a = b; b = temp; } while (0)

static double CPUcc_edge(const ECLgraph g, data_type* const label, const int* const sp, const int threadCount);

int cmp (const void * a, const void * b)
{
  if (*(double*)a < *(double*)b) {
    return -1;
  }
  else if (*(double*)a > *(double*)b) {
    return 1;
  }
  return 0;
}

static double median(double array[], const int n)
{
  double median = 0;
  qsort(array, n, sizeof(double), cmp);
  if (n % 2 == 0) median = (array[(n - 1) / 2] + array[n / 2]) / 2.0;
  else median = array[n / 2];
  return median;
}

int main(int argc, char* argv[])
{
  printf("cc edge-based C (%s)\n", __FILE__);
  if (argc != 3) {fprintf(stderr, "USAGE: %s input_file_name thread_count\n", argv[0]); exit(-1);}

  // process command line
  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  
  const int threadCount = atoi(argv[2]);
  printf("Threads: %d\n", threadCount);

  // create starting point array
  int* const sp = (int*)malloc(g.edges * sizeof(int));
  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      sp[j] = i;
    }
  }

  // allocate memory
  data_type* const label = (data_type*)malloc(g.nodes * sizeof(data_type));

  // cc
  const int runs = 3;
  double runtimes [runs];
  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPUcc_edge(g, label, sp, threadCount);
  }
  const double med = median(runtimes, runs);
  printf("C runtime: %.6fs\n", med);
  printf("C Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);

  // print result
  /*
  std::set<int> s1;
  for (int v = 0; v < g.nodes; v++) {
    s1.insert(label[v]);
  }
  printf("number of connected components: %d\n", s1.size());
  */
  
  // free memory
  free(label);
  free(sp);
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
