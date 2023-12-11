#include <sys/time.h>
#include <threads.h>
#include <stdatomic.h>
#include "ECLgraph.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define SWAP(a, b) do { __typeof__(a) temp = a; a = b; b = temp; } while (0)

static double CPUtc_vertex(basic_t* count, const int nodes, const int* const nindex, const int* const nlist, const int threadCount);

static inline int common(const int beg1, const int end1, const int beg2, const int end2, const int* const nlist)
{
  int common = 0;
  int pos1 = beg1;
  int pos2 = beg2;
  while ((pos1 < end1) && (pos2 < end2)) {
    while ((pos1 < end1) && (nlist[pos1] < nlist[pos2])) pos1++;
    if (pos1 < end1) {
      while ((pos2 < end2) && (nlist[pos2] < nlist[pos1])) pos2++;
      if ((pos2 < end2) && (nlist[pos1] == nlist[pos2])) {
        pos1++;
        pos2++;
        common++;
      } else {
        pos1++;
      }
    }
  }
  return common;
}

static inline int h_common(const int beg1, const int end1, const int beg2, const int end2, const int* const __restrict__ nlist)
{
  int common = 0;
  int pos1 = beg1;
  int pos2 = beg2;
  while ((pos1 < end1) && (pos2 < end2)) {
    while ((pos1 < end1) && (nlist[pos1] < nlist[pos2])) pos1++;
    if (pos1 < end1) {
      while ((pos2 < end2) && (nlist[pos2] < nlist[pos1])) pos2++;
      if ((pos2 < end2) && (nlist[pos1] == nlist[pos2])) {
        pos1++;
        pos2++;
        common++;
      } else {
        pos1++;
      }
    }
  }
  return common;
}

static basic_t h_triCounting(const int nodes, const int* const __restrict__ nindex, const int* const __restrict__ nlist)
{
  basic_t count = 0;

  for (int v = 0; v < nodes; v++) {
    const int beg1 = nindex[v];
    const int end1 = nindex[v + 1];
    int start1 = end1;
    while ((beg1 < start1) && (v < nlist[start1 - 1])) start1--;
    for (int j = start1; j < end1; j++) {
      const int u = nlist[j];
      const int beg2 = nindex[u];
      const int end2 = nindex[u + 1];
      int start2 = end2;
      while ((beg2 < start2) && (u < nlist[start2 - 1])) start2--;
      count += h_common(j + 1, end1, start2, end2, nlist);
    }
  }
  return count;
}

int cmp (const void * a, const void * b)
{
  if (*(double*)a < *(double*)b) {
    return -1;
  }
  else if (*(double*)a > *(double*)b) {
    return 1;
  }
  return 0;
}

static double median(double array[], const int n)
{
  double median = 0;
  qsort(array, n, sizeof(double), cmp);
  if (n % 2 == 0) median = (array[(n - 1) / 2] + array[n / 2]) / 2.0;
  else median = array[n / 2];
  return median;
}

int main(int argc, char* argv [])
{
  printf("Triangle counting vertex-centric C (%s)\n", __FILE__);

  if (argc != 3) {printf("USAGE: %s input_graph thread_count\n\n", argv[0]);  exit(-1);}

  // read input
  ECLgraph g = readECLgraph(argv[1]);
  printf("graph: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  printf("avg degree: %.2f\n", 1.0 * g.edges / g.nodes);

  // info only
  int mdeg = 0;
  for (int v = 0; v < g.nodes; v++) {
    mdeg = MAX(mdeg, g.nindex[v + 1] - g.nindex[v]);
  }
  printf("max degree: %d\n", mdeg);

  // check if sorted
  for (int v = 0; v < g.nodes; v++) {
    for (int i = g.nindex[v] + 1; i < g.nindex[v + 1]; i++) {
      if (g.nlist[i - 1] >= g.nlist[i]) {
        printf("ERROR: adjacency list not sorted or contains self edge\n");
        exit(-1);
      }
    }
  }
  
  const int threadCount = atoi(argv[2]);
  printf("Threads: %d\n\n", threadCount);

  // allocate memory
  basic_t count = 0;

  // launch kernel
  const int runs = 9;
  double runtimes [runs];

  for (int i = 0; i < runs; i++) {
    runtimes[i] = CPUtc_vertex(&count, g.nodes, g.nindex, g.nlist, threadCount);
  }

  const double med = median(runtimes, runs);
  printf("C runtime: %.6f s\n", med);
  printf("C Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);

  /*
  // verify
  const int verify = atoi(argv[2]);
  if ((verify != 0) && (verify != 1)) {
    printf("has to be 0 (turn off) or 1 (turn on) verification");
  }
  if (verify) {
    timeval start, end;
    gettimeofday(&start, NULL);
    */
    
    basic_t h_count = h_triCounting(g.nodes, g.nindex, g.nlist);
    
    //gettimeofday(&end, NULL);
    // printf("CPU runtime: %.6fs\n", end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0);
    if (h_count != count) printf("ERROR: host %ld device %ld", h_count, count);
    else printf("the pattern occurs %ld times\n\n", count);
  //}
  

  // clean up
  freeECLgraph(&g);
  return 0;
}
