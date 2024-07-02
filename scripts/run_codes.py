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

space = " "
log_dir = "run_logs/"
error_msg = 'USAGE: python3 ./' + os.path.basename(__file__) + ' exe_dir input_dir runs num_threads src write_to_file out_label\n\
\n\
exe_dir: directory containing executables to run\n\
input_dir: directory containing input graphs to use\n\
runs: the number of runs per input per program (reports median runtime)\n\
num_threads: the number of threads to use for C, CPP, OpenMP\n\
src: source vertex id for the BFS and SSSP codes\n\
write_to_file: 1 to write output to a log file in ./run_logs/\n\
out_label: labels log file ./run_logs/<out_label>_log.txt\n'

# create the run command
def create_run_cmd(exe_dir, exe_file, input, threads, runs, src):
    return os.path.join(exe_dir, exe_file) + space + input + space + runs + space + src + space + threads

def run_code(exe_dir, exe_file, input_files, num_inputs, input_dir, threads, runs, src, write_to_file, log_fin):
    if ("bfs" not in exe_file.lower()) and ("sssp" not in exe_file.lower()):
        src = ''
    input_counter = 0
    for input_file in input_files:
        input_counter += 1
        print("running %s on %s, %s out of %s inputs\n" % (exe_file, input_file, input_counter, num_inputs))
        if write_to_file == 1:
            os.system("echo " + ('"running %s on %s, %s out of %s inputs\n"' % (exe_file, input_file, input_counter, num_inputs)) + " >> " + log_file)
        run_cmd = create_run_cmd(exe_dir, exe_file, os.path.join(input_dir, input_file), threads, runs, src).split()
        #print(run_cmd)
        if write_to_file == 1:
            sys.stdout.flush()
            try:
                subprocess.run(run_cmd, env={"OMP_NUM_THREADS": threads}, stdout=log_fin, stderr=subprocess.STDOUT)
            except:
                log_fin.close()
                exit()
            log_fin.write('\n')
            log_fin.flush()
        else:
            sys.stdout.flush()
            try:
                subprocess.run(run_cmd, env={"OMP_NUM_THREADS": threads})
            except:
                exit()
            sys.stdout.write('\n')
            sys.stdout.flush()

if __name__ == "__main__":
    # read command line
    args_val = sys.argv
    if (len(args_val) != 8):
        sys.exit(error_msg)
    # @exe_dir: the directory of the executables
    # @input_dir: the directory of the input
    # @runs: the number of runs
    # @num_threads: number of threads
    # @src: source vertex id for bfs and sssp
    # @write_to_file: 0 or 1
    # @out_label: log file label

    # read inputs
    exe_dir = args_val[1]
    input_dir = args_val[2]
    runs = args_val[3]
    num_threads = args_val[4]
    src = args_val[5]
    write_to_file = int(args_val[6])
    out_label = args_val[7]
    
    # list code files
    exe_files = [f for f in os.listdir(exe_dir) if os.path.isfile(os.path.join(exe_dir, f))]
    num_codes = len(exe_files)
    print("exe_dir: %s\nnum_exes: %d" % (exe_dir, num_codes))

    # list input files
    input_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    num_inputs = len(input_files)
    print("input_dir: %s\nnum_inputs: %d\n" % (input_dir, num_inputs))

    # create log directory if requested
    log_file = ""
    log_fin = ""
    if write_to_file == 1:
        log_dir = os.path.join("./", log_dir)
        os.makedirs(log_dir, exist_ok=True)
        log_name = out_label + "_log.txt"
        log_file = os.path.join(log_dir, log_name)
        log_fin = open(log_file, "a")
        print(f"writing to log file {log_file}\n")

    # run the programs
    exe_counter = 0
    for exe_file in exe_files:
        exe_counter += 1
        run_code(exe_dir, exe_file, input_files, num_inputs, input_dir, num_threads, runs, src, write_to_file, log_fin)
    
    if write_to_file == 1:
        log_fin.close()


