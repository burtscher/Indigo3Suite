#include "graphGenerator.h"
#include <sys/stat.h>
/* Generate all the possible graphs with k vertices */

int main(int argc, char* argv[])
{
  // process the command line
  if (argc != 2) {fprintf(stderr, "USAGE: %s number_of_vertices\n", argv[0]); exit(-1);}
  const int n = atoi(argv[1]);
  if (n > 5) {fprintf(stderr, "ERROR: at most 5 vertices\n"); exit(-1);}
  if (n < 1) {fprintf(stderr, "ERROR: at least 1 vertex\n"); exit(-1);}
  const char* outpath = "./generatedGraphs";
  #ifdef __linux__
    mkdir(outpath, 0700);
  #else
    mkdir(outpath);
  #endif

  // the largest decimal to represent the graph matrices
  const int c = 1 << n * (n - 1);

  // generate every possible graph
  for (int i = 0; i < c; i++) {
    int val = i;
    int e = 0;
    std::set<int>* const edges = new std::set<int> [n];
    for (int row = 0; row < n; row++) {
      for (int col = 0; col < n; col++) {
        if (row != col) {
          if (val % 2) {
            edges[row].insert(col);
            e++;
          }
           val /= 2;
         }
      }
    }

    bool symmetric = true;
    for (int j = 0; j < n; j++) {
      for (int dst: edges[j]) {
        bool flag = true;
        for (int src: edges[dst]) {
          if (src == j) {
            flag = false;
            break;
          }
        }
        if (flag) {
          symmetric = false;
        }
      }
    }

    // save the graph in CSR format, print the histogram
    if (symmetric) {
      char name[256];
      sprintf(name, "%s/undirected_all_possible_graph_%d_%dn_%de.egr", outpath, i, n, e);
      printf("undirected_graph\n");
      saveAndPrint(n, e, name, edges);
    }
    delete [] edges;
  }
  return 0;
}