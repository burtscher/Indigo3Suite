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

exe_name = "minibench"
space = " "
log_file = "log.txt"
out_dir = "out/"
threads = "64"
omp_threads = "16"
error_msg = 'USAGE: code_dir input_dir lib_dir model gpu_computability runs src write_to_file out_label\n\
\n\
code_dir: relative path to directory containing generated codes to run\n\
input_dir: relative path to directory containing input graphs to use\n\
lib_dir: relative path to directory containing required header files (the lib folder)\n\
model:\n\
  0 for OpenMP\n\
  1 for C threads\n\
  2 for CUDA\n\
  3 for CPP threads\n\
gpu_computability: Compute capability of targeted GPU, decimal point removed\n\
runs: the number of runs per input per code\n\
src: source vertex id for the BFS and SSSP codes\n\
write_to_file: 1 to write output to a log file\n\
out_label: if write_to_file is 1, names log file: ./out/<out_label>_log.txt\n'

# compute the compile command
def compile_cmd(arch_number, model, lib_path, code_file_name):
    compiler = ["gcc", "gcc", "nvcc", "g++"]
    optimize_flag = "-O3"
    parallel_flag = ["-fopenmp", "-pthread -std=c11", "-DSLOWER_ATOMIC -arch=sm_" + arch_number, "-pthread -std=c++11"]
    library = "-I" + lib_path
    out_name = "-o" + space + exe_name
    return compiler[model] + space + optimize_flag + space + parallel_flag[model] + space + library + space + out_name + space + code_file_name

# compute the run command
def run_cmd(input, runs, src):
    return "./" + exe_name + space + input + space + runs + space + src + space + threads

def set_env(model):
    if (model == 0):
        os.system("export OMP_NUM_THREADS=" + omp_threads)

def compile_code(code_file, code_counter, num_codes, command):
    sys.stdout.flush()
    print("compile %s, %s out of %s programs\n" % (code_file, code_counter, num_codes))
    sys.stdout.flush()
    os.system(command)

def run_code(code_file, input_files, num_inputs, input_path, runs, src, write_to_file, log_file):
    if ("bfs" not in code_file.lower()) and ("sssp" not in code_file.lower()):
        src = ''
    input_counter = 0
    for input_file in input_files:
        input_counter += 1
        print("running %s on %s, %s out of %s inputs\n" % (code_file, input_file, input_counter, num_inputs))
        if write_to_file == 1:
            os.system("echo " + ('"running %s on %s, %s out of %s inputs\n"' % (code_file, input_file, input_counter, num_inputs)) + " >> " + log_file)
        set_env(model)
        run = run_cmd(os.path.join(input_path, input_file), runs, src)
        #print(run)
        if write_to_file == 1:
            sys.stdout.flush()
            os.system(run + " >> " + log_file + " 2>&1")
            os.system("echo >> " + log_file)
            sys.stdout.flush()
        else:
            sys.stdout.flush()
            os.system(run)
            sys.stdout.write('\n')
            sys.stdout.flush()

def create_out_dir(out_dir):
    if not os.path.exists(out_dir):
        print("create directory %s\n" % out_dir)
        os.mkdir(out_dir)
    else:
        print("directory %s already exists\n" % out_dir)

def delete_exe(exe_name):
    if os.path.isfile(exe_name):
        os.system("rm " + exe_name)
    else:
        sys.exit('Error: compile failed')

if __name__ == "__main__":
    # read command line
    args_val = sys.argv
    if (len(args_val) != 10):
        sys.exit(error_msg)
    # @code_dir: the directory of the code
    # @input_dir: the directory of the input
    # @lib_path: the directory of the library
    # @model: 
    #   0 for omp, 
    #   1 for C threads, 
    #   2 for cuda,
    #   3 for CPP threads
    # @gpu_computability: GPU computability
    # @runs: the number of runs
    # @src: source vertex id for bfs and sssp

    # read inputs
    code_path = args_val[1]
    input_path = args_val[2]
    lib_path = args_val[3]
    model = int(args_val[4])
    gpu_computability = args_val[5]
    runs = args_val[6]
    src = args_val[7]
    write_to_file = int(args_val[8])
    out_label = args_val[9]

    # list code files
    code_files = [f for f in os.listdir(code_path) if os.path.isfile(os.path.join(code_path, f))]
    num_codes = len(code_files)

    # list input files
    input_files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    num_inputs = len(input_files)
    print("num_codes: %d\nnum_inputs: %d\n" % (num_codes, num_inputs))
    print("code_path: %s\ninput_path: %s\n" % (code_path, input_path))


    # create output directory
    if write_to_file == 1: 
        create_out_dir(out_dir)
        current_directory = "./"
        out_dir = os.path.join(current_directory, out_dir)
        log_file = out_dir + out_label + "_log.txt"
        print("created out directory.out_path %s\nlog_file %s\n" % (out_dir, log_file))

    model_name = [".c", ".c", ".cu", ".cpp"]
    # compile and run the codes
    code_counter = 0
    for code_file in code_files:
        if code_file.endswith(model_name[model]):
            code_counter += 1
            compile_code(code_file, code_counter, num_codes, compile_cmd(gpu_computability, model, lib_path, os.path.join(code_path, code_file)))
            run_code(code_file, input_files, num_inputs, input_path, runs, src, write_to_file, log_file)
            delete_exe(exe_name)
        else:
            sys.exit('File %s does not match the programming model %s.' % (code_file, model_name[model]))


