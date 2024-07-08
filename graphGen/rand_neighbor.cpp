#include "graphGenerator.h"
#include <sys/stat.h>
/* Generate an undirected graph with n vertices and n edges where each vertex 
has one outgoing edge but the destination of the edge is random (with a uniform distribution) */

int main(int argc, char* argv[])
{
  // process the command line
  if (argc < 3) {fprintf(stderr, "USAGE: %s number_of_vertices random_seed\n", argv[0]); exit(-1);}
  const int n = atoi(argv[1]);
  if (n < 2) {fprintf(stderr, "ERROR: need at least 2 vertices\n"); exit(-1);}
  const int m = n;
  const int seed = atoi(argv[2]);
  const char* outpath = "./generatedGraphs";
  #ifdef __linux__
    mkdir(outpath, 0700);
  #else
    mkdir(outpath);
  #endif

  // generate random edges with uniform distribution of destination
  std::default_random_engine generator;
  std::uniform_int_distribution<int> distribution(0, n - 1);
  std::set<int>* const edges3 = new std::set<int> [n];
  for (int i = 0; i < n; i++) {
    int src, dst;
    src = i;
    do {
      dst = distribution(generator);
    } while (src == dst);
    edges3[src].insert(dst);
    edges3[dst].insert(src);
  }

  printf("\nUndirected random neighbors\n");
  char name3[256];
  sprintf(name3, "%s/undirect_rand_neighbors_%dn_%de.egr", outpath, n, m * 2);
  saveAndPrint(n, m * 2, name3, edges3);
  
  delete [] edges3;

  return 0;
}