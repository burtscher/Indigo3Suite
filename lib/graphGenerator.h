/*
This file is part of the Indigo3 benchmark suite version 1.0.

BSD 3-Clause License

Copyright (c) 2024, Yiqian Liu, Noushin Azami, Avery Vanausdal, and Martin Burtscher.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

URL: The latest version of the Indigo3 benchmark suite is available at https://github.com/burtscher/Indigo3Suite/.

Publication: This work is described in detail in the following paper.
Yiqian Liu, Noushin Azami, Avery Vanausdal, and Martin Burtscher. "Indigo3: A Parallel Graph Analytics Benchmark Suite for Exploring Implementation Styles and Common Bugs." ACM Transactions on Parallel Computing. May 2024.
*/


#include <stdio.h>
#include <random>
#include <set>
#include <algorithm>
#include "ECLgraph.h"

static ECLgraph toCSR(const int n, const int m, std::set<int>* edges)
{
  // convert to CSR graph format
  ECLgraph g;
  g.nodes = n;
  g.edges = m;
  g.nindex = new int [g.nodes + 1];
  g.nlist = new int [g.edges];
  g.eweight = NULL;
  int ecnt = 0;
  g.nindex[0] = 0;
  for (int i = 0; i < g.nodes; i++) {
    for (int dst: edges[i]) {
      g.nlist[ecnt] = dst;
      ecnt++;
    }
    g.nindex[i + 1] = ecnt;
  }
  return g;
}

static void printHistogram(const ECLgraph g)
{
  printf("number of vertices: %d\n", g.nodes);
  printf("number of edges: %d\n", g.edges);

  // count and sort the in-degree and out-degree
  int* const in_deg = new int [g.nodes];
  int* const out_deg = new int [g.nodes];
  for (int i = 0; i < g.nodes; i++) {
    in_deg[i] = 0;
  }
  for (int i = 0; i < g.nodes; i++) {
    out_deg[i] = g.nindex[i + 1] - g.nindex[i];
    const int beg = g.nindex[i];
    const int end = g.nindex[i + 1];
    for (int j = beg; j < end; j++) {
      const int v = g.nlist[j];
      in_deg[v]++;
    }
  }
  std::sort(in_deg, in_deg + g.nodes);
  std::sort(out_deg, out_deg + g.nodes);

  int d = in_deg[0];
  int count = 0;
  printf("=====in-degree histogram=====\n");
  printf("in-degree          frequency\n");
  for (int i = 0; i < g.nodes; i++) {
    if (d < in_deg[i]) {
      printf("%d%27d\n", d, count);
      d = in_deg[i];
      count = 1;
    } else if (d == in_deg[i]) {
      count++;
    }
  }
  printf("%d%27d\n", d, count);

  d = out_deg[0];
  count = 0;
  printf("=====out-degree histogram=====\n");
  printf("out-degree          frequency\n");
  for (int i = 0; i < g.nodes; i++) {
    if (d < out_deg[i]) {
      printf("%d%27d\n", d, count);
      d = out_deg[i];
      count = 1;
    } else if (d == out_deg[i]) {
      count++;
    }
  }
  printf("%d%27d\n", d, count);

  delete [] in_deg;
  delete [] out_deg;
  return;
}

static void saveAndPrint(const int n, const int m, const char* const fname, std::set<int>* edges)
{
  ECLgraph g = toCSR(n, m, edges);
  writeECLgraph(g, fname);
  printHistogram(g);
  freeECLgraph(&g);
}