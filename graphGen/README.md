# Graph Generator

This directory contains a selection of undirected graph generators for your convenience.

The /inputs/ folder includes all 64 possible undirected graphs with 4 vertices and a few examples from each of the other generators. 

To compile any graph generator: 

    g++ -I../lib -O3 <graph_generator>.cpp -o graphGenerator

To use a graph generator, run it to discover the parameters:

    ./graphGenerator

The generators with a "random_seed" parameter will generate 1 graph for that seed. Try multiple seeds if you don't get the result you're looking for.

If a graph is listed with twice as many edges as expected, this is due to the added reverse edges that make the graph undirected.

Generated graphs will go into the `./generatedGraphs/` subdirectory.
