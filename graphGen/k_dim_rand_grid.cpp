#include "graphGenerator.h"
#include <sys/stat.h>
/* Generate a k-dimensional directed grid, the size of each dimension is decided
by the input, and the vertex IDs are shuffled randomly */

int main(int argc, char* argv[])
{
  // process the command line
  if (argc < 6) {fprintf(stderr, "USAGE: %s min_vertices max_vertices dim dim_size random_seed\n", argv[0]); exit(-1);}
  const int minV = atoi(argv[1]);
  const int maxV = atoi(argv[2]);
  const int dim = atoi(argv[3]);
  if (dim < 1) {fprintf(stderr, "ERROR: need at least 1 dimension\n"); exit(-1);}
  const int size = atoi(argv[4]);
  if (size < 1) {fprintf(stderr, "ERROR: width has to be larger than 1\n"); exit(-1);}
  const int seed = atoi(argv[5]);
  const char* outpath = "./generatedGraphs";
  #ifdef __linux__
    mkdir(outpath, 0700);
  #else
    mkdir(outpath);
  #endif

  // precompute multipliers
  int* const mult = new int [dim];
  int n = 1;
  for (int i = 0; i < dim; i++) {
    mult[i] = n;
    n *= size;
  }

  if (n >= minV && n <= maxV) {
    // create a random map to shuffle the vertex IDs
    int* const map = new int [n];
    for (int i = 0; i < n; i++) {
      map[i] = i;
    }
    std::mt19937 gen(seed);
    shuffle(map, map + n, gen);

    std::set<int>* const edges1 = new std::set<int> [n];
    std::set<int>* const edges2 = new std::set<int> [n];
    std::set<int>* const edges3 = new std::set<int> [n];
    int m1 = 0;
    int m2 = 0;

    // generate edges
    int* const coords = new int [dim + 1];
    for (int i = 0; i <= dim; i++) coords[i] = 0;
    do {
      int src = 0;
      for (int i = 0; i < dim; i++) {
        src += coords[i] * mult[i];
      }
      for (int i = 0; i < dim; i++) {
        int dst = src - coords[i] * mult[i];
        if (coords[i] + 1 < size) {
          dst += ((coords[i] + 1) % size) * mult[i];
          edges1[map[src]].insert(map[dst]);
          edges2[map[dst]].insert(map[src]);
          m1++;
          if (edges3[map[src]].find(map[dst]) == edges3[map[src]].end()) {
            edges3[map[src]].insert(map[dst]);
            edges3[map[dst]].insert(map[src]);
            m2++;
          }
        }
      }

      int i = -1;
      do {
        i++;
        coords[i] = (coords[i] + 1) % size;
      } while (coords[i] == 0);
    } while (coords[dim] == 0);

    // save the graph
    printf("\nUndirected %d-dimensional random grid\n", dim);
    char name3[256];
    sprintf(name3, "%s/undirect%ddim_rand_grid_%dn_%de.egr", outpath, dim, n, m2 * 2);
    saveAndPrint(n, m2 * 2, name3, edges3);
    
    delete [] edges1;
    delete [] edges2;
    delete [] edges3;
  } else {
    printf("N=%i out of range\n", n);
  }
  
  delete [] mult;
  return 0;
}