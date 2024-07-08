#include "graphGenerator.h"
#include <sys/stat.h>
/* Generate an undirected graph with n vertices, n-1 edges, and a random vertex r 
where each vertex has one outgoing edge and the destination of the edge is always r 
(vertex r has no outgoing edge) */

int main(int argc, char* argv[])
{
  // process the command line
  if (argc < 3) {fprintf(stderr, "USAGE: %s number_of_vertices random_seed\n", argv[0]); exit(-1);}
  const int n = atoi(argv[1]);
  if (n < 2) {fprintf(stderr, "ERROR: need at least 2 vertices\n"); exit(-1);}
  const int m = n - 1;
  const int seed = atoi(argv[2]);
  const char* outpath = "./generatedGraphs";
  #ifdef __linux__
    mkdir(outpath, 0700);
  #else
    mkdir(outpath);
  #endif

  // randomly generate a destination for all (n - 1) edges
  std::set<int>* const edges1 = new std::set<int> [n];
  std::set<int>* const edges2 = new std::set<int> [n];
  std::set<int>* const edges3 = new std::set<int> [n];
  srand(seed);
  const int dst = rand() % n;
  for (int i = 0; i < n; i++) {
    const int src = i;
    if (i != dst) {
      edges1[src].insert(dst);
      edges2[dst].insert(src);
      edges3[src].insert(dst);
      edges3[dst].insert(src);
    }
  }
  
  printf("\nUndirected star\n");
  char name3[256];
  sprintf(name3, "%s/undirect_star_%dn_%de.egr", outpath, n, m * 2);
  saveAndPrint(n, m * 2, name3, edges3);
  
  delete [] edges1;
  delete [] edges2;
  delete [] edges3;

  return 0;
}
