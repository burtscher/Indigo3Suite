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
LIABLE FOR ANY DIRECT, exe_dirECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
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


import sys
import os
import re
import subprocess

error_msg = 'USAGE: ./' + os.path.basename(__file__) + ' input_dir runs num_threads src write_to_file programming_model(optional) algorithm(optional)\n\
\n\
input_dir: directory containing input graphs to use\n\
runs: the number of runs per input per program (reports median runtime)\n\
num_threads: the number of threads to use for C, CPP, OpenMP\n\
src: source vertex id for the BFS and SSSP codes\n\
write_to_file: 1 to write output to log files in ./run_logs/\n\
\n\
programming_model: ALL, C, CPP, OMP, or CUDA (case insensitive) default=ALL\n\
algorithm: ALL, BFS, CC, MIS, MST, PR, SSSP, or TC (case insensitive) default=ALL\n'

base_path = "./executables/"
all_models = ["C", "CPP", "OMP", "CUDA"]
all_codes = ["BFS", "CC", "MIS", "MST", "PR", "SSSP", "TC"]

args = sys.argv
if len(args) < 6:
    sys.exit(error_msg)

input_dir = args[1]
runs = args[2]
num_threads = args[3]
src = args[4]
write_to_file = args[5]

model_arg = "ALL"
if len(args) > 6:
    model_arg = args[6].upper()

codes_arg = "ALL"
if len(args) > 7:
    codes_arg = args[7].upper()

models = [model_arg]
if model_arg == "ALL":
    models = all_models
elif model_arg not in all_models:
    print("ERROR: Invalid programming_model argument")
    sys.exit(error_msg)

codes = [codes_arg]
if codes_arg == "ALL":
    codes = all_codes
elif codes_arg not in all_codes:
    print("ERROR: Invalid algorithm argument")
    sys.exit(error_msg)

#check num_threads argument
if ("C" in models or "CPP" in models or "OMP" in models) and not num_threads.isdigit():
    print("ERROR: Invalid num_threads argument, specify a number")
    sys.exit(error_msg)

#check src argument
if ("BFS" in codes or "SSSP" in codes) and not src.isdigit():
    print("ERROR: Invalid src argument, specify a number")
    sys.exit(error_msg)
    
print(f"Running {codes_arg} codes for {model_arg} model(s) on inputs from {input_dir}\n")

for model in models:
    for code in codes:
        folder_name = code + '-' + model
        exe_dir = os.path.join(os.path.join(base_path, model), folder_name)
        if not os.path.isdir(exe_dir):
            print(f"{exe_dir} not found, skipping...")
            continue
        
        subprocess.run(["python3", "./scripts/run_codes.py", exe_dir, input_dir, runs, num_threads, src, write_to_file, folder_name])
