#include <limits.h>
#include <sys/time.h>
#include <threads.h>
#include <stdatomic.h>
#include <stdbool.h>
#include "ECLgraph.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define SWAP(a, b) do { __typeof__(a) temp = a; a = b; b = temp; } while (0)

//const data_type maxval = std::numeric_limits<data_type>::max();

static double CPPbfs_edge(const int src, const ECLgraph g, data_type* const dist, const int* const sp, const int threadCount);

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
  printf("bfs edge-based C (%s)\n", __FILE__);
  if (argc != 5) {fprintf(stderr, "USAGE: %s input_file_name source_node_number runs thread_count\n", argv[0]); exit(-1);}

  // process command line
  ECLgraph g = readECLgraph(argv[1]);

  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  const int source = atoi(argv[2]);
  if ((source < 0) || (source >= g.nodes)) {fprintf(stderr, "ERROR: source_node_number must be between 0 and %d\n", g.nodes); exit(-1);}
  printf("source: %d\n", source);
  const int runs = atoi(argv[3]);

  const int threadCount = atoi(argv[4]);
  printf("Threads: %d\n", threadCount);

  // create starting point array
  int* const sp = (int*)malloc(g.edges * sizeof(int));
  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      sp[j] = i;
    }
  }

  // allocate memory
  data_type* const distance = (data_type*)malloc(g.nodes * sizeof(data_type));
  double runtimes [runs];
  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPPbfs_edge(source, g, distance, sp, threadCount);
  }
  const double med = median(runtimes, runs);
  printf("runtime: %.6fs\n", med);
  printf("Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);

  // print result
  int maxnode = 0;
  for (int v = 1; v < g.nodes; v++) {
    if (distance[maxnode] < distance[v]) maxnode = v;
  }
  printf("vertex %d has maximum distance %d\n", maxnode, distance[maxnode]);

  // free memory
  free(distance);
  free(sp);
  freeECLgraph(&g);
  return 0;
}

static inline data_type atomicMin(shared_t* addr, data_type val)
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
