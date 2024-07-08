#include "graphGenerator.h"
#include <sys/stat.h>
/* Generate disconnected undirected binary trees with n vertices */

int main(int argc, char* argv[])
{
  // process the command line
  if (argc < 3) {fprintf(stderr, "USAGE: %s number_of_vertices random_seed\n", argv[0]); exit(-1);}
  const int n = atoi(argv[1]);
  if (n < 2) {fprintf(stderr, "ERROR: need at least 2 vertices\n"); exit(-1);}
  int m = 0;
  const int seed = atoi(argv[2]);
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

  std::set<int>* const edges3 = new std::set<int> [n];
  srand(seed);
  const int p = 5;
  int count = 0;
  for (int i = 0; i < n && count < n; i++) {
    bool left = rand() % p;
    bool right = rand() % p;
    int src = i;
    int dst;

    if (left > 0) {
      dst = count + 1;
      if (dst < n) {
        if (edges3[map[src]].find(map[dst]) == edges3[map[src]].end()) {
          edges3[map[src]].insert(map[dst]);
          edges3[map[dst]].insert(map[src]);
          m++;
        }
        count = dst;
      }
    }
    if (right > 0) {
      dst = count + 1;
      if (dst < n) {
        edges3[map[src]].insert(map[dst]);
        edges3[map[dst]].insert(map[src]);
        m++;
        count = dst;
      }
    }
    if (i == count) {
      count++;
    }
  }
  printf("\nUndirected binary forest\n");
  char name3[256];
  sprintf(name3, "%s/undirect_binary_forest_%dn_%de.egr", outpath, n, m * 2);
  saveAndPrint(n, m * 2, name3, edges3);

  delete [] edges3;

  return 0;
}