# Indigo3Suite

Indigo3 is a labeled benchmark suite based on 7 graph algorithms that are implemented in different styles, including versions with deliberately planted bugs.

## Generating codes

To generate all codes for all models, run the generate script with no arguments:

    `python3 ./generate_all_codes.py`

The script has 2 optional parameters for specifying a specific model and/or code.

For example, this command will generate all of the OpenMP codes:

    `python3 ./generate_all_codes.py omp`
    
While this command will generate the Connected Components codes for all models:

    `python3 ./generate_all_codes.py all cc`

The generated codes will be in the `generatedCodes` directory. You can also modify `configure.txt` under the `codeGen` directory to define the type of bugs you want to generate.

## Compiling codes

To compile all codes for all models from the `generatedCodes` directory, run the compile script with only the gpu_computability parameter:

    `python3 ./compile_all_codes.py <gpu_computability>`
    
The compile script has the same optional parameters as the generate script. For example, this command will compile just the Breadth-First Search OpenMP codes:

    `python3 ./generate_all_codes.py 0 omp bfs`

The gpu_computability parameter specifies the targeted NVIDIA GPU compute capability for CUDA codes. For example, to compile all CUDA codes for a Titan V, which has a compute capability of 7.0, use the following:

    `python3 ./generate_all_codes.py 70 cuda`

The compiled executables will be in the `executables` directory.

## Running codes

A `./run_all_codes.py` script is also provided. It looks in the `executables` directory. To see the full list of parameters, run it without arguments:

    `python3 ./run_all_codes.py`
    
For example, this command will run the compiled Connected Components C codes on the inputs in `./inputs/` using 32 CPU threads and writing the output to `./run_logs/`:

    `python3 ./run_all_codes.py ./inputs/ 1 32 0 1`

## Inputs

Small sample inputs are available in the `inputs` directory. The `download_large_inputs.sh` script will download five additional large graphs and place them in a `large_inputs` directory.

The codes in this suite use ECL graphs stored in binary CSR format. Converters and additional inputs are available [here](https://userweb.cs.txstate.edu/~burtscher/research/ECLgraph/).

## Citing Indigo3

If you use Indigo3, please cite the following publication.

* Yiqian Liu, Noushin Azami, Avery Vanausdal, and Martin Burtscher. "Indigo3: A Parallel Graph Analytics Benchmark Suite for Exploring Implementation Styles and Common Bugs." ACM Transactions on Parallel Computing. May 2024.
[[doi]](https://doi.org/10.1145/3665251)
[[pdf]](https://userweb.cs.txstate.edu/~burtscher/papers/topc24.pdf)

You may also be interested in the predecessor suites, [Indigo](https://cs.txstate.edu/~burtscher/research/IndigoSuite/) and [Indigo2](https://cs.txstate.edu/~burtscher/research/Indigo2Suite/).

*This work has been supported in part by the National Science Foundation under Grant No. 1955367 as well as by an equipment donation from NVIDIA Corporation.*
