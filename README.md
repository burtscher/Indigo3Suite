# Indigo3Suite

Indigo3 is a labeled benchmark suite based on 7 parallel graph algorithms that are implemented in different styles, including versions with deliberately planted bugs. Each algorithm is implemented in parallel for C, C++, OpenMP, CUDA, and HIP.

## Generating codes

To generate ALL codes for ALL models, run the generate script with no arguments:

    python3 ./generate_all_codes.py

The script has 2 optional parameters for specifying specific models and/or codes. The model options are C, CPP, OMP, CUDA, HIP, and ALL. The code options are BFS, CC, MIS, MST, PR, SSSP, TC, and ALL. Both parameters are case-insensitive, can accept a comma-separated list of options, and default to ALL.

For example, this command will generate all of the C and OpenMP codes:

    python3 ./generate_all_codes.py c,omp
    
While this command will generate the Connected Components codes for all models:

    python3 ./generate_all_codes.py all cc

The generated codes will be in the `generatedCodes/` directory. 

You can also modify `codeGen/configure.txt` to enable bug styles and define a subset of styles you want to generate.

## Compiling codes

To compile HIP codes for AMD GPUs, refer to the [HIP Prerequisite for AMD GPUs](#hip-prerequisite-for-amd-gpus) section.

To compile all codes for all CPU models in the `generatedCodes/` directory, run the compile script with no parameters:

    python3 ./compile_all_codes.py

The compile script has the same parameters for specifying codes and models as the generate script. For example, this command will compile just the Breadth-First Search C++ codes:

    python3 ./compile_all_codes.py cpp bfs

To compile CUDA or HIP codes for NVIDIA GPUs, the nvidia_compute_capability parameter is required. The nvidia_compute_capability parameter specifies the targeted NVIDIA GPU [compute capability](https://developer.nvidia.com/cuda-gpus). For example, to compile all CUDA codes for a Titan V, which has a compute capability of 7.0, use the following:

    python3 ./compile_all_codes.py cuda all 70

The compiled executables will be in the `executables/` directory.

## Running codes

A `./run_all_codes.py` script is also provided. It looks in the `executables/` directory. To see the full list of parameters, run it without arguments:

    python3 ./run_all_codes.py
    
For example, this command will run the compiled Connected Components C codes in `executables/` on the inputs in `inputs/` using 32 CPU threads and write the output to `run_logs/`:

    python3 ./run_all_codes.py inputs/ 1 32 0 1 c cc

## Inputs

Small sample inputs are available in the `inputs/` directory. The `download_large_inputs.sh` script will download five additional large graphs and place them in a `large_inputs/` directory. Graph generators for creating additional small inputs are provided in the `graphGen/` directory with their own README.

The codes in this suite use ECL graphs stored in binary CSR format. Converters and additional inputs are available [here](https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/).

## HIP Prerequisite for AMD GPUs

[libhipcxx](https://github.com/ROCm/libhipcxx) is required to compile the HIP codes for AMD GPUs. It does not need to be built. The default location for libhipcxx is in the same directory as `Indigo3Suite/` (not inside it). If `libhipcxx` is installed elsewhere, you will need to edit the `libhipcxx_path` variable in `scripts/compile_codes.py` to point to your `libhipcxx` directory.

## Citing Indigo3

If you use Indigo3, please cite the following publication.

* Yiqian Liu, Noushin Azami, Avery Vanausdal, and Martin Burtscher. "Indigo3: A Parallel Graph Analytics Benchmark Suite for Exploring Implementation Styles and Common Bugs." ACM Transactions on Parallel Computing. May 2024.
[[doi]](https://doi.org/10.1145/3665251)
[[pdf]](https://userweb.cs.txstate.edu/~burtscher/papers/topc24.pdf)

You may also be interested in the predecessor suites [Indigo](https://cs.txstate.edu/~burtscher/research/IndigoSuite/) and [Indigo2](https://cs.txstate.edu/~burtscher/research/Indigo2Suite/) as well as in the related [ECL-Suite](https://github.com/burtscher/ECL-Suite/).

The following paper describes ideas on how to use Indigo3 (aka Sapphire) for teaching parallel programming.

* Yiqian Liu, Noushin Azami, Avery Vanausdal, and Martin Burtscher. "Sapphire: a Tool for Teaching Parallel Programming in Hundreds of Different Ways." Proceedings of the 16th Annual International Conference on Education and New Learning Technologies. July 2024.
[[doi]](https://doi.org/10.21125/edulearn.2024.1136)
[[pdf]](https://userweb.cs.txstate.edu/~burtscher/papers/edulearn24.pdf)
[[pptx]](https://userweb.cs.txstate.edu/~burtscher/slides/edulearn24.pptx)

The following paper describes the process and performance implications of porting the CUDA codes in Indigo3 to HIP.

* Avery Vanausdal and Martin Burtscher. "Comparing Graph Algorithm Styles on NVIDIA and AMD GPUs." Proceedings of the 15th SC Workshop on Irregular Applications: Architectures and Algorithms. November 2025.
[[doi]](https://dl.acm.org/doi/10.1145/3731599.3767444)
[[pdf]](https://userweb.cs.txstate.edu/~burtscher/papers/ia25b.pdf)

*This work has been supported in part by the National Science Foundation under Grant No. 1955367 as well as by an equipment donation from NVIDIA Corporation.*
