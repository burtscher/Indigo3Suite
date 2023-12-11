#include <limits.h>
#include <sys/time.h>
#include <cuda.h>
#include "ECLgraph.h"
#include "cuda_atomic.h"
#include "csort.h"
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define SWAP(a, b) do { __typeof__(a) temp = a; a = b; b = temp; } while (0)

const basic_t maxval = INT_MAX;

static double GPUsssp_edge(const int src, const ECLgraph g, basic_t* const dist, const int* const sp);

static int GPUinfo(const int d)
{
  cudaSetDevice(d);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, d);
  if ((deviceProp.major == 9999) && (deviceProp.minor == 9999)) {printf("ERROR: there is no CUDA capable device\n\n");  exit(-1);}
  const int mTpSM = deviceProp.maxThreadsPerMultiProcessor;
  const int SMs = deviceProp.multiProcessorCount;
  printf("GPU: %s with %d SMs and %d mTpSM (%.1f MHz and %.1f MHz)\n", deviceProp.name, SMs, mTpSM, deviceProp.clockRate * 0.001, deviceProp.memoryClockRate * 0.001);
  return SMs * mTpSM;
}

static void CheckCuda()
{
  cudaError_t e;
  cudaDeviceSynchronize();
  if (cudaSuccess != (e = cudaGetLastError())) {
    fprintf(stderr, "CUDA error %d: %s\n", e, cudaGetErrorString(e));
    exit(-1);
  }
}

int main(int argc, char* argv[])
{
  printf("(%s)\n", __FILE__);
  if (argc < 4) {fprintf(stderr, "USAGE: %s input_file_name runs srouce\n", argv[0]); exit(-1);}

  // process command line
  ECLgraph g = readECLgraph(argv[1]);
  if (g.eweight == NULL) {
    printf("Generating weights.\n");
    g.eweight = (data_type*)malloc(g.edges * sizeof(data_type));
    for (int i = 0; i < g.nodes; i++) {
      for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
        const int nei = g.nlist[j];
        g.eweight[j] = 1 + ((i * nei) % g.nodes);
          if (g.eweight[j] < 0) g.eweight[j] = -g.eweight[j];
      }
    }
  }
  printf("input: %s\n", argv[1]);
  printf("nodes: %d\n", g.nodes);
  printf("edges: %d\n", g.edges);
  const int source = atoi(argv[3]);
  if ((source < 0) || (source >= g.nodes)) {fprintf(stderr, "ERROR: source_node_number must be between 0 and %d\n", g.nodes); exit(-1);}
  printf("source: %d\n", source);

  // create starting point array
  int* const sp = (int*)malloc(g.edges * sizeof(int));
  for (int i = 0; i < g.nodes; i++) {
    for (int j = g.nindex[i]; j < g.nindex[i + 1]; j++) {
      sp[j] = i;
    }
  }

  // allocate memory
  basic_t* const distance = (basic_t*)malloc(g.nodes * sizeof(basic_t));
  ECLgraph d_g = g;
  if (cudaSuccess != cudaMalloc((void **)&d_g.nindex, (g.nodes + 1) * sizeof(int))) fprintf(stderr, "ERROR: could not allocate nindex\n");
  if (cudaSuccess != cudaMalloc((void **)&d_g.nlist, g.edges * sizeof(int))) fprintf(stderr, "ERROR: could not allocate nlist\n");
  if (cudaSuccess != cudaMalloc((void **)&d_g.eweight, g.edges * sizeof(int))) fprintf(stderr, "ERROR: could not allocate eweight\n");
  if (cudaSuccess != cudaMemcpy(d_g.nindex, g.nindex, (g.nodes + 1) * sizeof(int), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of index to device failed\n");
  if (cudaSuccess != cudaMemcpy(d_g.nlist, g.nlist, g.edges * sizeof(int), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of nlist to device failed\n");
  if (cudaSuccess != cudaMemcpy(d_g.eweight, g.eweight, g.edges * sizeof(int), cudaMemcpyHostToDevice)) fprintf(stderr, "ERROR: copying of eweight to device failed\n");

  const int runs = atoi(argv[2]);
  double runtimes [runs];

  for (int i = 0; i < runs; i++) {
    runtimes[i] = GPUsssp_edge(source, d_g, distance, sp);
    CheckCuda();
  }
  const double med = median(runtimes, runs);
  printf("GPU runtime: %.6f s\n", med);
  printf("GPU Throughput: %.6f gigaedges/s\n", 0.000000001 * g.edges / med);

  // free memory
  free(distance);
  free(sp);
  cudaFree(d_g.nindex);
  cudaFree(d_g.nlist);
  cudaFree(d_g.eweight);
  freeECLgraph(&g);
  return 0;
}
