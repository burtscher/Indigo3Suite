#include <sys/time.h>
#include <math.h>
#include "ECLgraph.h"
#include "csort.h"
#include "omp_helper.h"

static const data_type EPSILON = 0.0001;
static const data_type kDamp = 0.85;
static const int ThreadsPerBlock = 256;
static const int MAX_ITER = 100;

void PR_CPU(const ECLgraph g, data_type *scores, int* degree);

int main(int argc, char *argv[]) {
  printf("PageRank OMP v0.1 (%s)\n", __FILE__);
  printf("Copyright 2022 Texas State University\n\n");

  if (argc < 3) {printf("USAGE: %s input_graph runs\n\n", argv[0]);  exit(-1);}

  // read input
  ECLgraph g = readECLgraph(argv[1]);
  printf("graph: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  printf("avg degree: %.2f\n", 1.0 * g.edges / g.nodes);

  // count degree
  int* degree = (int*)malloc(g.nodes * sizeof(int));
  for (int i = 0; i < g.nodes; i++) {
    degree[i] = g.nindex[i + 1] - g.nindex[i];
  }

  // init scores
  const data_type init_score = 1.0f / (data_type)g.nodes;
  data_type* scores = (data_type*)malloc(g.nodes * sizeof(data_type));
  for (int i = 0; i < g.nodes; i++) {
    scores[i] = init_score;
  }

  PR_CPU(g, scores, degree);
  
  // compare and verify
  const data_type base_score = (1.0f - kDamp) / (data_type)g.nodes;
  data_type* incoming_sums = (data_type*)malloc(g.nodes * sizeof(data_type));
  for(int i = 0; i < g.nodes; i++) incoming_sums[i] = 0;
  double error = 0;
  
  for (int src = 0; src < g.nodes; src++) {
    data_type outgoing = scores[src] / degree[src];
    for (int i = g.nindex[src]; i < g.nindex[src + 1]; i++) {
      incoming_sums[g.nlist[i]] += outgoing;
    }
  }
  
  for (int i = 0; i < g.nodes; i++) {
    data_type new_score = base_score + kDamp * incoming_sums[i];
    error += fabs(new_score - scores[i]);
    incoming_sums[i] = 0;
  }
  if (error < EPSILON) printf("All good.\n");
  else printf("Total Error: %f\n", error);
  
  // free memory
  free(degree);
  free(scores);
  free(incoming_sums);
  freeECLgraph(&g);
  return 0;
}
