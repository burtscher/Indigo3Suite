#include <sys/time.h>
#include <stdbool.h>
#include <limits.h>
#include "ECLgraph.h"
#include "csort.h"
#include "omp_helper.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define SWAP(a, b) do { __typeof__(a) temp = a; a = b; b = temp; } while (0)

const int maxval = INT_MAX;

static double CPUmst(const ECLgraph g, const int* const sp, bool* const included);

int main(int argc, char* argv[])
{
  printf("MST edge-based OMP (%s)\n", __FILE__);
  if (argc < 3) {fprintf(stderr, "USAGE: %s input_file_name runs thread_count(optional)\n", argv[0]); exit(-1);}

  // process command line
  ECLgraph g = readECLgraph(argv[1]);
  if (g.eweight == NULL) {
    printf("Generating weights.\n");
    g.eweight = malloc(sizeof(int) * g.edges);
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
  const int runs = atoi(argv[2]);
  
  // create starting point array
  int* const sp = (int*)malloc(g.edges * sizeof(int));
  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      sp[j] = i;
    }
  }

  // allocate memory
  bool* const included = (bool*)malloc(g.edges * sizeof(bool));

  // launch kernel
  double runtimes [runs];
  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPUmst(g, sp, included);
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
  free(sp);
  free(included);
  freeECLgraph(&g);
  return 0;
}

static void updateFromWorklist(unsigned char* const status, unsigned char* const status_n, const int* const worklist, const int wlsize)
{
  #pragma omp parallel for
  for (int i = 0; i < wlsize; ++i)
  {
    int v = worklist[i];
    status[v] = status_n[v];
  }
}

static inline int critical_CAS(int* addr, int compare, int val)
{
  int oldv;
  #pragma omp critical
  {
    oldv = *addr;
    if (oldv == compare) {
      *addr = val;
    }
  }
  return oldv;
}
