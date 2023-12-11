#include <sys/time.h>
#include <limits.h>
#include <stdbool.h>
#include "ECLgraph.h"
#include "csort.h"
#include "omp_helper.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define SWAP(a, b) do { typeof(a) temp = a; a = b; b = temp; } while (0)


static double CPUcc_vertex(const ECLgraph g, data_type* label);

int main(int argc, char* argv[])
{
  printf("cc topology-driven OMP (%s)\n", __FILE__);
  if (argc < 3) {fprintf(stderr, "USAGE: %s input_file_name runs\n", argv[0]); exit(-1);}

  // process command line
  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);

  // allocate memory
  data_type* const label = (data_type*)malloc(g.nodes * sizeof(data_type));

  // cc
  const int runs = atoi(argv[2]);
  double runtimes [runs];
  bool flag = true;
  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPUcc_vertex(g, label);
  }
  const double med = median(runtimes, runs);
  printf("runtime: %.6fs\n", med);
  printf("Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);

  // free memory
  free(label);
  freeECLgraph(&g);
  return 0;
}
