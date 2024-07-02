# Indigo3Suite

Indigo3 is a labeled benchmark suite based on 7 graph algorithms that are implemented in different styles, including versions with deliberately planted bugs.

For example, you can create Breadth-First Search OpenMP codes as follows:

`python3 scripts/generate_codes.py codeGen/omp/BFS-OMP/BFS_OMP_V_Topo_Push.idg bfs-omp omp`

The generated codes will be in the `bfs-omp` directory.

You can generate all of the OpenMP codes as follows:

`python3 ./generate_all_codes omp`

You can also modify the `configure.txt` under the `codeGen` directory to define the type of bugs you want to generate.



You may also be interested in the predecessor suites, [Indigo](https://cs.txstate.edu/~burtscher/research/IndigoSuite/) and [Indigo2](https://cs.txstate.edu/~burtscher/research/Indigo2Suite/).

*This work has been supported in part by the National Science Foundation under Grant No. 1955367 as well as by an equipment donation from NVIDIA Corporation.*
