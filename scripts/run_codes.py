#!/usr/bin/python3 -u

import os
import sys

# verifiers
# 0: clang static analyzer
#   compile: clang --analyze -Xclang -analyzer-checker=core minibench.c
# 1: CIVL
#   compile: java -jar /home/yiqian/verifiers/CIVL-trunk_5834/lib/civl-trunk_5834.jar verify minibench
# 2: Archer
#   compile: clang-archer example.c -o example
#   run: ./minibench runs src
# 3: tSan
#    compile: clang -fopenmp -fsanitize=thread -O3 -o minibench minibench.c
#    run: TSAN_OPTIONS="suppressions=tsan_suppress.txt" ./minibench runs threads src
# 4: compute-sanitizer
#    compile: nvcc -O3 -arch=sm_35 -Ilib -o minibench minibench.cu
#    run: compute-sanitizer --tool memcheck ./minibench
#        
# 5: compute-sanitizer --tool racecheck ./minibench
# 6: compute-sanitizer --tool initcheck ./minibench
# 7: compute-sanitizer --tool synccheck ./minibench
# 8: iGuard
#    compile: nvcc -O3 -arch=sm_35 -Ilib -o minibench minibench.cu
#    run: LD_PRELOAD=./nvbit_release/tools/detector/detector.so ./minibench
# 9: faial
#   compile: ./faial-drf
 
run_verify_cmd = ["",
                  "java -jar /home/yiqian/verifiers/CIVL-trunk_5834/lib/civl-trunk_5834.jar verify", 
                  "",
                  "",
                   "compute-sanitizer --tool memcheck",
                   "compute-sanitizer --tool racecheck",
                   "compute-sanitizer --tool initcheck",
                   "compute-sanitizer --tool synccheck",
                   "LD_PRELOAD=./../verifiers/iGUARD-SOSP21/nvbit_release/tools/detector/detector.so",
                   "./faial-drf"]

exe_name = "minibench"
space = " "
log_file = "log.txt"
out_dir = "out/"
threads = "64"

# check if the verify id
def isAnalyzer(verifier_id):
    return verifier_id == 0

def isCIVL(verifier_id):
    return verifier_id == 1

def isTsan(verifier_id):
    return verifier_id == 3

def isFaial(verifier_id):
    return verifier_id == 9

# compute the compile command
def compile_cmd(arch_number, model, lib_path, verifiers, code_file_name):
    if isCIVL(verifiers) or isFaial(verifiers):
        return run_verify_cmd[verifiers] + space + code_file_name
    compiler = ["clang --analyze -Xclang -analyzer-checker=core", "", "clang-archer", "clang -fsanitize=thread", "nvcc", "nvcc", "nvcc", "nvcc", "nvcc"]
    optimize_flag = "-O3"
    parallel_flag = ["-fopenmp", "-pthread -std=c11", "-DSLOWER_ATOMIC -arch=sm_" + arch_number]
    library = "-I" + lib_path
    out_name = "-o" + space + exe_name
    return compiler[verifiers] + space + optimize_flag + space + parallel_flag[model] + space + library + space + out_name + space + code_file_name

# compute the run command
def run_cmd(verifiers, input, runs, src):
    return run_verify_cmd[verifiers] + space + "./" + exe_name + space + input + space + runs + space + src + space + threads

def set_env(verifier_id, model):
    if (isTsan(verifier_id)):
        os.system("export TSAN_OPTIONS='ignore_noninstrumented_modules=1'")
    if (model == 0):
        os.system("export OMP_NUM_THREADS=16")

def compile_code(code_file, code_counter, num_codes, command):
    sys.stdout.flush()
    print("compile %s, %s out of %s programs\n" % (code_file, code_counter, num_codes))
    sys.stdout.flush()
    os.system(command)

def run_code(code_file, input_files, num_inputs, verifier_id, input_path, runs, src, write_to_file, log_file):
    input_counter = 0
    for input_file in input_files:
        input_counter += 1
        print("running %s, %s out of %s inputs\n" % (code_file, input_counter, num_inputs))
        set_env(verifier_id, model)
        run = run_cmd(verifier_id, os.path.join(input_path, input_file), runs, src)
        if write_to_file == 1:
            with open(log_file, 'a') as f:         
                sys.stdout.flush()
                os.system(run + " > " + log_file)
                sys.stdout.flush()
        else:
            sys.stdout.flush()
            os.system(run)
            sys.stdout.flush()

def create_out_dir(out_dir):
    if not os.path.exists(out_dir):
        print("create directory %s\n" % out_dir)
        os.mkdir(out_dir)
    else:
        print("directory %s already exists\n" % out_dir)

def run_test(verifier_id, only_compile):
    return (not isCIVL(verifier_id)) and (not isFaial(verifier_id)) and (not isAnalyzer(verifier_id)) and (only_compile != 1)

def delete_exe(exe_name):
    if os.path.isfile(exe_name):
        os.system("rm " + exe_name)
    else:
        sys.exit('Error: compile failed')

if __name__ == "__main__":
    # read command line
    args_val = sys.argv
    if (len(args_val) != 12):
        sys.exit('USAGE: code_dir input_dir lib_path model verifiers gpu_computability runs src only_compile write_to_file out_label\n')
    # @code_dir: the directory of the code
    # @input_dir: the directory of the input
    # @lib_path: the directory of the library
    # @model: 
    #   0 for omp, 
    #   1 for pthread, 
    #   2 for cuda
    # @verifiers: 
    #   0 for clang static analyzer
    #   1 for CIVL
    #   2 for Archer
    #   3 for tSan
    #   4 for compute-sanitizer memcheck
    #   5 for compute-sanitizer racecheck
    #   6 for compute-sanitizer initcheck
    #   7 for iGuard
    #   8 for faial
    # @computability: GPU computability
    # @runs: the number of runs
    # @src: source vertex id for bfs, sssp, and cc

    # read inputs
    code_path = args_val[1]
    input_path = args_val[2]
    lib_path = args_val[3]
    model = int(args_val[4])
    verifier_id = int(args_val[5])
    gpu_computability = args_val[6]
    runs = args_val[7]
    src = args_val[8]
    only_compile = int(args_val[9])
    write_to_file = int(args_val[10])
    out_lable = args_val[11]

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
        current_directory = "/home/yiqian/Indigo2Verification"
        out_dir = os.path.join(current_directory, out_dir)
        log_file = out_dir + out_lable + "_log.txt"
        print("creat out directory.out_path %s\nlog_file %s\n" % (out_dir, log_file))

    model_name = [".c", ".c", ".cu"]
    # compile and run the codes
    code_counter = 0
    for code_file in code_files:
        if code_file.endswith(model_name[model]):
            code_counter += 1
            compile_code(code_file, code_counter, num_codes, compile_cmd(gpu_computability, model, lib_path, verifier_id, os.path.join(code_path, code_file)))
            if run_test(verifier_id, only_compile):
                run_code(code_file, input_files, num_inputs, verifier_id, input_path, runs, src, write_to_file, log_file)
            delete_exe(exe_name)
            # if os.path.isfile(exe_name):
            #     os.system("rm " + exe_name)
            # else:
            #     sys.exit('Error: compile failed')
        else:
            sys.exit('File %s does not match the programming model %s.' % (code_file, model_name[model]))

    
    # # compile and run minibenchmark
    # counter = 0
    # for code_file in code_files:
    #     for input_file in input_files:
    #         if code_file.endswith(model_name[model]):
    #             counter += 1
    #             sys.stdout.flush()
    #             print("testing %s\n" % (code_file))
    #             set_env(verifier_id, model)
    #             sys.stdout.flush()
    #             with open(log_file, 'a') as f:
    #                         file_path = os.path.join(code_path, code_file)
    #                         sys.stdout.flush()
    #                         f.write('\ncompile : %s\n' % code_file)
    #                         sys.stdout.flush()
    #                         compile = compile_cmd(gpu_computability, model, lib_path, verifier_id, file_path)
    #                         run = run_cmd(verifier_id, input_path, runs, src)
    #                         print(compile)
    #                         print(run)
    #                         os.system(compile)
    #                         if not isCIVL(verifier_id) and not isFaial(verifier_id) and not isAnalyzer(verifier_id):
    #                             os.system(run + " >> " + log_file)
    #                         sys.stdout.flush()
    #                         if os.path.isfile(exe_name):
    #                             os.system("rm " + exe_name)
    #                         else:
    #                             sys.exit('Error: compile failed')
    #         else:
    #             sys.exit('No codes in the directory.')
