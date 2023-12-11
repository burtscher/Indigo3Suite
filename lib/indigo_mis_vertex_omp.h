#include <sys/time.h>
#include <limits.h>
#include <stdbool.h>
#include "ECLgraph.h"
#include "csort.h"
#include "omp_helper.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define SWAP(a, b) do { typeof(a) temp = a; a = b; b = temp; } while (0)

const data_type maxval = INT_MAX;

const data_type undecided = 0;
const data_type included = 1;
const data_type excluded = 2;

static double OMPmis_vertex(const ECLgraph g, data_type* const priority, data_type* const status);

int main(int argc, char* argv[])
{
  printf("mis vertex-based OMP (%s)\n", __FILE__);
  if (argc < 3) {fprintf(stderr, "USAGE: %s input_file_name runs\n", argv[0]); exit(-1);}

  // process command line
  ECLgraph g = readECLgraph(argv[1]);
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);

  // allocate memory
  data_type* const priority = (data_type*)malloc(g.nodes * sizeof(data_type));
  data_type* const status = (data_type*)malloc(g.nodes * sizeof(data_type));

  // launch kernel
  const int runs = atoi(argv[2]);
  double runtimes [runs];
  for (int i = 0; i < runs; i++) {
    runtimes[i] = OMPmis_vertex(g, priority, status);
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

static void updateUndecided(data_type* const status, data_type* const status_n, const int size)
{
  #pragma omp parallel for
  for (int i = 0; i < size; ++i)
  {
    if (status[i] == undecided)
      status[i] = status_n[i];
  }
}

static void updateFromWorklist(data_type* const status, data_type* const status_n, const int* const worklist, const int wlsize)
{
  #pragma omp parallel for
  for (int i = 0; i < wlsize; ++i)
  {
    int v = worklist[i];
    status[v] = status_n[v];
  }
}
