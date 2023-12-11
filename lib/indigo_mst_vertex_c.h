#include <sys/time.h>
#include <threads.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <limits.h>
#include "ECLgraph.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define SWAP(a, b) do { __typeof__(a) temp = a; a = b; b = temp; } while (0)

static double CPUmst(const ECLgraph g, bool* const included, const int threadCount);

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
  printf("MST vertex-based C (%s)\n", __FILE__);
  if (argc != 3) {fprintf(stderr, "USAGE: %s input_file_name thread_count\n", argv[0]); exit(-1);}

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
  
  const int threadCount = atoi(argv[2]);
  printf("Threads: %d\n", threadCount);

  // allocate memory
  bool* const included = (bool*)malloc(g.edges * sizeof(bool));

  // launch kernel
  const int runs = 9;
  double runtimes [runs];
  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPUmst(g, included, threadCount);
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
  printf("MST includes %d edges with %d weight\n", count, weight);

  // free memory
  free(included);
  freeECLgraph(&g);
  return 0;
}

static inline int atomicCAS(_Atomic int* addr, int compare, int val)
{
  atomic_compare_exchange_strong(addr, &compare, val);
  return compare;
}

static inline int atomicCAS_cast(int* addr, int compare, int val)
{
  atomic_compare_exchange_strong((_Atomic int*)addr, &compare, val);
  return compare;
}

static inline int atomicMax(_Atomic int* addr, int val)
{
  int oldv = atomic_load(addr);
  while (oldv < val && !(atomic_compare_exchange_weak(addr, &oldv, val))) {}
  return oldv;
}
