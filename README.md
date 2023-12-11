# Indigo3Suite

Indigo3 is a labeled benchmark suite based on 7 graph algorithms that are implemented in different styles, including versions with deliberately planted bugs.

For example, you can create Breadth-First Search OpenMP codes as follows:

`python scripts/generate_codes.py codeGen/omp/BFS-OMP/BFS_OMP_V_Topo_Push.idg bfs-omp c`

The generated codes will be in the `bfs-omp` directory, and all the codes will feature the `.c` extension, reflecting the C-style programming.

You can also modify the  `configure.txt` under the `codeGen` directory to define the type of bugs you want to generate.
