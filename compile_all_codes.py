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

error_msg = 'USAGE: ./' + os.path.basename(__file__) + ' programming_models(default=ALLCPU) algorithms(default=ALL) nvidia_compute_capability(optional for non-NVIDIA)\n\
\n\
programming_models: C, CPP, OMP, CUDA, HIP-AMD, HIP-NVIDIA, and ALLCPU (case insensitive, comma separated) default=ALLCPU\n\
algorithms: BFS, CC, MIS, MST, PR, SSSP, TC, and ALL (case insensitive, comma separated) default=ALL\n\
nvidia_compute_capability: Compute capability of targeted NVIDIA GPU, without decimal point (optional for non-NVIDIA)\n'

base_path = "./generatedCodes/"
base_outdir = "./executables/"
all_models = ["C", "CPP", "OMP", "CUDA", "HIP-AMD", "HIP-NVIDIA"]
all_CPU = ["C", "CPP", "OMP"]
all_codes = ["BFS", "CC", "MIS", "MST", "PR", "SSSP", "TC"]

args = sys.argv
# if len(args) < 2:
    # sys.exit(error_msg)

model_arg = "ALLCPU"
if len(args) > 1:
    model_arg = args[1].upper()

codes_arg = "ALL"
if len(args) > 2:
    codes_arg = args[2].upper()

nvidia_compute_capability = None
if len(args) > 3:
    nvidia_compute_capability = args[3]

models = set(model_arg.split(','))
if "ALLCPU" in model_arg:
    models.remove("ALLCPU")
    models.update(all_CPU)
else:
    for model in models:
        if model not in all_models:
          print("ERROR: Invalid programming_model argument:", model)
          sys.exit(error_msg)

codes = set(codes_arg.split(','))
if "ALL" in codes_arg:
    codes = all_codes
else:
    for code in codes:
        if code not in all_codes:
          print("ERROR: Invalid algorithm argument:", code)
          sys.exit(error_msg)

#if CUDA, check nvidia_compute_capability argument
if ('CUDA' in models or 'HIP-NVIDIA' in models) and (not nvidia_compute_capability or not nvidia_compute_capability.isdigit()):
    print("ERROR: Targeting NVIDIA but nvidia_compute_capability argument is missing or invalid, specify a number")
    sys.exit(error_msg)
    
print(f"Compiling {', '.join(codes)} codes for {', '.join(models)} model(s)\n")

for model in models:
    for code in codes:
        short_model = model.split('-')[0] # Remove -AMD and -NVIDIA from HIP model name to match directory name
        folder_name = code + '-' + short_model
        indir = os.path.join(os.path.join(base_path, short_model), folder_name)
        if not os.path.isdir(indir):
            print(f"{indir} not found, skipping...")
            continue
        
        outdir = os.path.join(os.path.join(base_outdir, model), folder_name)
        os.makedirs(outdir, exist_ok=True)
        
        run_cmd = ["python3", "./scripts/compile_codes.py", indir, outdir, model]
        if nvidia_compute_capability:
            run_cmd.append(nvidia_compute_capability)
        
        subprocess.run(run_cmd)
