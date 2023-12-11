#include <sys/time.h>
#include <stdbool.h>
#include "ECLgraph.h"
#include "csort.h"
#include "omp_helper.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

static double CPUtc_vertex(data_type* count, const int nodes, const int* const nindex, const int* const nlist);

static inline int common(const int beg1, const int end1, const int beg2, const int end2, const int* const nlist)
{
  int common = 0;
  int pos1 = beg1;
  int pos2 = beg2;
  while ((pos1 < end1) && (pos2 < end2)) {
    while ((pos1 < end1) && (nlist[pos1] < nlist[pos2])) pos1++;
    if (pos1 < end1) {
      while ((pos2 < end2) && (nlist[pos2] < nlist[pos1])) pos2++;
      if ((pos2 < end2) && (nlist[pos1] == nlist[pos2])) {
        pos1++;
        pos2++;
        common++;
      } else {
        pos1++;
      }
    }
  }
  return common;
}

int main(int argc, char* argv [])
{
  printf("Triangle counting vertex-centric (%s)\n", __FILE__);

  if (argc < 3) {printf("USAGE: %s input_graph runs\n\n", argv[0]);  exit(-1);}

  // read input
  ECLgraph g = readECLgraph(argv[1]);
  printf("graph: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  printf("avg degree: %.2f\n", 1.0 * g.edges / g.nodes);

  // info only
  int mdeg = 0;
  for (int v = 0; v < g.nodes; v++) {
    mdeg = MAX(mdeg, g.nindex[v + 1] - g.nindex[v]);
  }
  printf("max degree: %d\n\n", mdeg);

  // check if sorted
  for (int v = 0; v < g.nodes; v++) {
    for (int i = g.nindex[v] + 1; i < g.nindex[v + 1]; i++) {
      if (g.nlist[i - 1] >= g.nlist[i]) {
        printf("ERROR: adjacency list not sorted or contains self edge\n");
        exit(-1);
      }
    }
  }

  // allocate memory
  data_type* count = malloc(sizeof(data_type));
  *count = 0;

  // launch kernel
  const int runs = atoi(argv[2]);
  double runtimes [runs];

  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPUtc_vertex(count, g.nodes, g.nindex, g.nlist);
  }

  const double med = median(runtimes, runs);
  printf("OMP runtime: %.6f s\n", med);
  printf("OMP Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);

  // clean up
  freeECLgraph(&g);
  return 0;
}
