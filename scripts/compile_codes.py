#!/usr/bin/python3 -u

'''
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
'''


import os
import sys
import subprocess

lib_defaultpath = "lib/"
error_msg = 'USAGE: python3 ./' + os.path.basename(__file__) + ' code_dir output_dir programming_model gpu_computability lib_dir(optional)\n\
\n\
code_dir: directory containing generated codes to compile\n\
output_dir: name of directory to place compiled executables in (will be created)\n\
programming_model: C, CPP, OMP, or CUDA (case insensitive)\n\
gpu_computability: Compute capability of targeted GPU, without decimal point (for CUDA)\n\
\n\
lib_dir: library directory, only necessary if working directory is not repsitory root\n'
model_map = {"omp" : 0, "c" : 1, "cuda" : 2, "cpp" : 3}

# compute the compile command
def compile_cmd(arch_number, model, lib_path, code_file_name, out_dir):
    compilers = ["gcc", "gcc", "nvcc", "g++"]
    optimize_flag = "-O3"
    parallel_flags = ["-fopenmp", "-pthread -std=c11", "-arch=sm_" + arch_number, "-pthread -std=c++11"] #-DSLOWER_ATOMIC disabled
    library = "-I" + lib_path
    exe_name = os.path.splitext(os.path.basename(code_file_name))[0]
    out_name = os.path.join(out_dir, exe_name)
    return [compilers[model], optimize_flag, *parallel_flags[model].split(), library, "-o", out_name, code_file_name]

def compile_code(code_file, code_counter, num_codes, command):
    sys.stdout.flush()
    print("compile %s, %s out of %s programs\n" % (code_file, code_counter, num_codes))
    sys.stdout.flush()
    try:
        subprocess.run(command)
    except:
        exit()

if __name__ == "__main__":
    # read command line
    args_val = sys.argv
    if (len(args_val) < 5):
        sys.exit(error_msg)
    # @code_dir: the directory of the code
    # @output_dir: the directory where the executables go
    # @programming_model: C, CPP, OMP, or CUDA
    # @gpu_computability: GPU computability
    # @lib_path (optional): the directory of the library

    # read inputs
    code_path = args_val[1]
    out_dir = args_val[2]
    model_arg = args_val[3].lower()
    gpu_computability = args_val[4]
    
    if os.path.isdir(lib_defaultpath):
        lib_path = lib_defaultpath
    elif len(args_val) >= 6 and os.path.isdir(args_val[5]):
        lib_path = args_val[5]
    else:
        print("ERROR: lib/ directory not found, specify location in arguments:", file=sys.stderr)
        sys.exit(error_msg)
    
    out_dir = os.path.join(os.getcwd(), out_dir)
    os.makedirs(out_dir, exist_ok=True)
    
    model = -1
    if model_arg in model_map:
        model = model_map[model_arg]
    else:
        print("ERROR: Invalid programming_model argument, specify one of the following: C, CPP, OMP, CUDA\n", file=sys.stderr)
        sys.exit(error_msg)

    # list code files
    code_files = [f for f in os.listdir(code_path) if os.path.isfile(os.path.join(code_path, f))]
    num_codes = len(code_files)
    print("code_dir: %s" % (code_path))
    print("num_codes: %d\n" % (num_codes))

    model_name = [".c", ".c", ".cu", ".cpp"]
    # compile the codes
    code_counter = 0
    for code_file in code_files:
        if code_file.endswith(model_name[model]):
            code_counter += 1
            compile_code(code_file, code_counter, num_codes, compile_cmd(gpu_computability, model, lib_path, os.path.join(code_path, code_file), out_dir))
        else:
            sys.exit('File %s does not match the programming model %s.' % (code_file, model_name[model]))


