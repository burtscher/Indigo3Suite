#include "graphGenerator.h"
#include <sys/stat.h>
/* Generate undirected binary trees with n vertices */

struct Node
{
  int key;
  int id;
  Node* left;
  Node* right;
};

Node* newNode(int k, int v)
{
  Node* newnode = new Node;
  newnode->key = k;
  newnode->id = v;
  newnode->left = newnode->right = NULL;
  return newnode;
}

Node* insert(Node* root, int k, int v)
{
  Node* newnode = newNode(k, v);
  Node* cur = root;
  Node* y = NULL;
  while (cur != NULL) {
    y = cur;
    if (k < cur->key) {
      cur = cur->left;
    } else {
      cur = cur->right;
    }
  }

  if (y == NULL) {
    y = newnode;
  } else if (k < y->key) {
    y->left = newnode;
  } else {
    y->right = newnode;
  }
  return y;
}

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

  std::set<int>* const edges1 = new std::set<int> [n];
  std::set<int>* const edges2 = new std::set<int> [n];
  std::set<int>* const edges3 = new std::set<int> [n];
  Node* root = NULL;
  root = insert(root, map[0], 0);
  for (int i = 1; i < n; i++) {
    Node* node = insert(root, map[i], i);
    int src = node->id;
    int dst = i;
    int parent_id = node->id;
    edges1[src].insert(dst);
    edges2[dst].insert(src);
    edges3[src].insert(dst);
    edges3[dst].insert(src);
    m++;
  }
  printf("\nUndirected binary tree\n");
  char name3[256];
  sprintf(name3, "%s/undirect_binary_tree_%dn_%de.egr", outpath, n, m * 2);
  saveAndPrint(n, m * 2, name3, edges3);
  
  delete [] edges1;
  delete [] edges2;
  delete [] edges3;

  return 0;
}