#include "graphGenerator.h"
#include <sys/stat.h>
/* Generate an undirected graph with n vertices and m edges (with a uniform distribution) */

int main(int argc, char* argv[])
{
  // process the command line
  if (argc < 4) {fprintf(stderr, "USAGE: %s number_of_vertices number_of_edges random_seed\n", argv[0]); exit(-1);}
  const int n = atoi(argv[1]);
  if (n < 2) {fprintf(stderr, "ERROR: need at least 2 vertices\n"); exit(-1);}
  const int m = atoi(argv[2]);
  if ((m <= 0) || (m > n * (n - 1) / 2)) {fprintf(stderr, "ERROR: number of edges out of range\n"); exit(-1);}
  const int seed = atoi(argv[3]);
  const char* outpath = "./generatedGraphs";
  #ifdef __linux__
    mkdir(outpath, 0700);
  #else
    mkdir(outpath);
  #endif

  // create a random map to shuffle the vertex IDs
  int* const map = new int [n];
  for (int i = 0; i < n; i++) {
    map[i] = i;
  }
  std::mt19937 gen(seed);
  shuffle(map, map + n, gen);

  // generate random edges with uniform distribution of endpoints
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, n - 1);
  std::set<int>* const edges3 = new std::set<int> [n];
  for (int i = 0; i < m; i++) {
    int src, dst;
    do {
      src = distribution(generator);
      do {
        dst = distribution(generator);
      } while ((dst >= n) || (src == dst));
      src = map[src];
      dst = map[dst];
    } while (edges3[src].find(dst) != edges3[src].end());
    edges3[src].insert(dst);
    edges3[dst].insert(src);
  }

  printf("\nUndirected random uniform degree graph\n");
  char name3[256];
  sprintf(name3, "%s/undirect_uniform_degree_%dn_%de.egr", outpath, n, m * 2);
  saveAndPrint(n, m * 2, name3, edges3);

  delete [] edges3;

  return 0;
}
