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


import sys
import os
import re
import subprocess

error_msg = 'USAGE: ./' + os.path.basename(__file__) + ' gpu_compute_capability programming_model(optional) algorithm(optional)\n\
\n\
gpu_compute_capability: Compute capability of targeted GPU, without decimal point (for CUDA)\n\
\n\
programming_model: C, CPP, OMP, CUDA, or ALL (case insensitive) default=ALL\n\
algorithm: BFS, CC, MIS, MST, PR, SSSP, TC, or ALL (case insensitive) default=ALL\n'

base_path = "./generatedCodes/"
base_outdir = "./executables/"
all_models = ["C", "CPP", "OMP", "CUDA"]
all_codes = ["BFS", "CC", "MIS", "MST", "PR", "SSSP", "TC"]

args = sys.argv
if len(args) < 2:
    sys.exit(error_msg)

gpu_compute_capability = args[1]

model_arg = "ALL"
if len(args) > 2:
    model_arg = args[2].upper()

codes_arg = "ALL"
if len(args) > 3:
    codes_arg = args[3].upper()

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

#if CUDA, check gpu_compute_capability argument
if "CUDA" in models and not gpu_compute_capability.isdigit():
    print("ERROR: Invalid gpu_compute_capability argument, specify a number")
    sys.exit(error_msg)
    
print(f"Compiling {codes_arg} codes for {model_arg} model(s)\n")

for model in models:
    for code in codes:
        folder_name = code + '-' + model
        indir = os.path.join(os.path.join(base_path, model), folder_name)
        if not os.path.isdir(indir):
            print(f"{indir} not found, skipping...")
            continue
        
        outdir = os.path.join(os.path.join(base_outdir, model), folder_name)
        os.makedirs(outdir, exist_ok=True)
        
        subprocess.run(["python3", "./scripts/compile_codes.py", indir, outdir, model, gpu_compute_capability])
