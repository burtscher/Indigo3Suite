#include <sys/time.h>
#include <limits.h>
#include <stdbool.h>
#include "ECLgraph.h"
#include "csort.h"
#include "omp_helper.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define SWAP(a, b) do { typeof(a) temp = a; a = b; b = temp; } while (0)
const data_type maxval = INT_MAX;

static double CPUbfs_vertex(const int src, const ECLgraph g, data_type* dist);

int main(int argc, char* argv[])
{
  printf("bfs OMP (%s)\n", __FILE__);
  if (argc < 4) {fprintf(stderr, "USAGE: %s input_file_name runs source\n", argv[0]); exit(-1);}

  // process command line
  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  const int source = atoi(argv[3]);
  if ((source < 0) || (source >= g.nodes)) {fprintf(stderr, "ERROR: source_node_number must be between 0 and %d\n", g.nodes); exit(-1);}
  printf("source: %d\n", source);
  const int runs = atoi(argv[2]);

  // allocate memory
  data_type* const distance = (data_type*)malloc(g.nodes * sizeof(data_type));

  // bfs
  double runtimes [runs];
  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPUbfs_vertex(source, g, distance);
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

  free(distance);
  freeECLgraph(&g);
  return 0;
}
