#include "graphGenerator.h"
#include <sys/stat.h>
/* Generate an undirected graph with n vertices where the max in and out degree
is capped at k */

int main(int argc, char* argv[])
{
  // process the command line
  if (argc < 4) {fprintf(stderr, "USAGE: %s number_of_vertices max_degree random_seed\n", argv[0]); exit(-1);}
  const int n = atoi(argv[1]);
  if (n < 2) {fprintf(stderr, "ERROR: need at least 2 vertices\n"); exit(-1);}
  const int maxD = atoi(argv[2]);
  if (maxD < 1) {fprintf(stderr, "ERROR: maximum degree must be at least 1\n"); exit(-1);}
  const int seed = atoi(argv[3]);
  int m = 0;
  const char* outpath = "./generatedGraphs";
  #ifdef __linux__
    mkdir(outpath, 0700);
  #else
    mkdir(outpath);
  #endif

  srand(seed);
  std::set<int>* const edges1 = new std::set<int> [n];
  std::set<int>* const edges2 = new std::set<int> [n];
  std::set<int>* const edges3 = new std::set<int> [n];

  // generate edges
  for (int i = 0; i < n; i++) {
    int src, dst;
    src = i;
    for (int j = 0; j < maxD / 2; j++) {
      dst = rand() % n;
      if (src != dst && edges3[dst].size() < maxD && edges3[src].size() < maxD) {
        edges1[src].insert(dst);
        edges2[dst].insert(src);
        edges3[src].insert(dst);
        edges3[dst].insert(src);
      }
    }
    m += edges1[src].size();
  }

  printf("\nUndirected in and out\n");
  char name3[256];
  sprintf(name3, "%s/undirect_%dmax_degree_%dn_%de.egr", outpath, maxD, n, m * 2);
  saveAndPrint(n, m * 2, name3, edges3);

  delete [] edges1;
  delete [] edges2;
  delete [] edges3;

  return 0;
}
