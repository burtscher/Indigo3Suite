#include <algorithm>
#include <sys/time.h>
#include <math.h>
#include <thread>
#include <atomic>
#include "ECLgraph.h"

static const score_type EPSILON = 0.0001;
static const score_type kDamp = 0.85;
static const int MAX_ITER = 100;

static double PR_CPU(const ECLgraph g, score_type *scores, int* degree, const int threadCount);

static double median(double array[], const int n)
{
  double median = 0;
  std::sort(array, array + n);
  if (n % 2 == 0) median = (array[(n - 1) / 2] + array[n / 2]) / 2.0;
  else median = array[n / 2];
  return median;
}

template <typename T>
static inline T atomicAdd(std::atomic<T>* addr, T val)
{
  T old = addr->load();
  while (!(addr->compare_exchange_weak(old, old + val))) {}
  return old;
}

int main(int argc, char *argv[]) {
  printf("PageRank CPP v0.1 (%s)\n", __FILE__);
  printf("Copyright 2022 Texas State University\n\n");

  if (argc != 2 && argc != 3) {printf("USAGE: %s input_graph thread_count(optional)\n\n", argv[0]);  exit(-1);}

  // read input
  ECLgraph g = readECLgraph(argv[1]);
  printf("graph: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  printf("avg degree: %.2f\n", 1.0 * g.edges / g.nodes);

  // count degree
  int* degree = new int [g.nodes];
  for (int i = 0; i < g.nodes; i++) {
    degree[i] = g.nindex[i + 1] - g.nindex[i];
  }

  int threadCount = std::thread::hardware_concurrency(); //defaults to max threads
  if(argc == 3)
    if(const int countInt = atoi(argv[2])) //checks for valid int
      threadCount = countInt;             //takes optional argument for thread count
  printf("Threads: %d\n\n", threadCount);

  // init scores
  const score_type init_score = 1.0f / (score_type)g.nodes;
  score_type* scores = new score_type [g.nodes];
  std::fill(scores, scores + g.nodes, init_score);

  double runtime = PR_CPU(g, scores, degree, threadCount);
  
  printf("CPU runtime: %.6fs\n\n", runtime);
  printf("Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / runtime);
  
  // compare and verify
  const score_type base_score = (1.0f - kDamp) / (score_type)g.nodes;
  score_type* incoming_sums = new score_type [g.nodes];
  for(int i = 0; i < g.nodes; i++) incoming_sums[i] = 0;
  double error = 0;
  
  for (int src = 0; src < g.nodes; src++) {
    score_type outgoing = scores[src] / degree[src];
    for (int i = g.nindex[src]; i < g.nindex[src + 1]; i++) {
      incoming_sums[g.nlist[i]] += outgoing;
    }
  }
  
  for (int i = 0; i < g.nodes; i++) {
    score_type new_score = base_score + kDamp * incoming_sums[i];
    error += fabs(new_score - scores[i]);
    incoming_sums[i] = 0;
  }
  if (error < EPSILON) printf("All good.\n");
  else printf("Total Error: %f\n", error);
  
  // free memory
  delete [] degree;
  delete [] scores;
  delete [] incoming_sums;
  freeECLgraph(&g);
  return 0;
}
