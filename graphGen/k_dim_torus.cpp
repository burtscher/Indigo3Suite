#include "graphGenerator.h"
#include <sys/stat.h>
/* Generate a k-dimensional torus, the size of each dimension is decided by the input */

int main(int argc, char* argv[])
{
  // process the command line
  if (argc < 5) {fprintf(stderr, "USAGE: %s min_vertices max_vertices dim dim_size\n", argv[0]); exit(-1);}
  const int minV = atoi(argv[1]);
  const int maxV = atoi(argv[2]);
  const int dim = atoi(argv[3]);
  if (dim < 1) {fprintf(stderr, "ERROR: need at least 1 dimension\n"); exit(-1);}
  const int size = atoi(argv[4]);
  if (size < 1) {fprintf(stderr, "ERROR: width has to be larger than 1\n"); exit(-1);}
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
    // init
    std::set<int>* const edges3 = new std::set<int> [n];
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
        dst += ((coords[i] + 1) % size) * mult[i];
        if (edges3[src].find(dst) == edges3[src].end()) {
          edges3[src].insert(dst);
          edges3[dst].insert(src);
          m2++;
        }
      }

      int i = -1;
      do {
        i++;
        coords[i] = (coords[i] + 1) % size;
      } while (coords[i] == 0);
    } while (coords[dim] == 0);

    printf("\nUndirected %d-dimensional torus\n", dim);
    char name3[256];
    sprintf(name3, "%s/undirect%ddim_torus_%dn_%de.egr", outpath, dim, n, m2 * 2);
    saveAndPrint(n, m2 * 2, name3, edges3);
    
    delete [] edges3;
  } else {
    printf("N=%i out of range\n", n);
  }
  
  delete [] mult;
  return 0;
}