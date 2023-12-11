#include <sys/time.h>
#include <threads.h>
#include <stdatomic.h>
#include <stdbool.h>
#include "ECLgraph.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define SWAP(a, b) do { __typeof__(a) temp = a; a = b; b = temp; } while (0)

const unsigned char undecided = 0;
const unsigned char included = 1;
const unsigned char excluded = 2;

static double CPPmis_edge(const ECLgraph g, const int* const sp, data_type* const priority, unsigned char* const status, const int threadCount);

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
  printf("mis edge-based C (%s)\n", __FILE__);
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
  data_type* const priority = (data_type*)malloc(g.nodes * sizeof(data_type));
  unsigned char* const status = (unsigned char*)malloc(g.nodes * sizeof(unsigned char));
  
  const int runs = 3;
  double runtimes [runs];
  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPPmis_edge(g, sp, priority, status, threadCount);
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

static inline int atomicMax(_Atomic int* addr, int val)
{
  int oldv = atomic_load(addr);
  while (oldv < val && !(atomic_compare_exchange_weak(addr, &oldv, val))) {}
  return oldv;
}
