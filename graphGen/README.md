# Graph Generator

All 64 possible undirected graphs with 4 vertices are present in the /inputs/ folder. This generator can produce similar sets for different vertex counts.

To compile: 

    make

or

    g++ -I../lib -O3 --std=c++11 all_possible_graphs.cpp -o graphGenerator

To use the graph generator, specify the number of vertices the graphs should have:

    ./graphGenerator <num_of_vertices>

There are 2n(n-1)<sup>2</sup> possible graphs with n vertices, so 5 vertices will generate 1024 files. Enter higher values at your own risk.