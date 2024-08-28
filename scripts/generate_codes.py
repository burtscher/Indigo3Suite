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


import itertools
import sys
import os
import re

copyright_defaultpath = "copyright.txt"
config_defaultpath = "codeGen/configure.txt"
error_msg = 'USAGE: python3 ./' + os.path.basename(__file__) + ' input_file output_dir programming_model config_file(optional) copyright_file(optional)\n\
\n\
input_file: .idg file to generate codes from\n\
output_dir: name of directory to place generated codes in (will be created)\n\
programming_model: C, CPP, OMP, or CUDA (case insensitive)\n\
\n\
config_file: location of configure.txt, only necessary if working directory is not repository root\n\
copyright_file: location of copyright.txt, only necessary if working directory is not repository root\n'
model_map = {"omp" : ".c", "c" : ".c", "cuda" : ".cu", "cpp" : ".cpp"}

def needBugComment(o_tag, o_code):
	tag = o_tag.lower()
	code = o_code.lower()
	if ("race" in tag and "no" not in tag and "atomic" not in code and code) or ("bug" in tag and "no" not in tag and code): # if Race or Bug
		return True
	else:
		return False

# filter data type
def filter_datat(idg_file, code_file, dataTypes):
	isKeeper = False
	datat = ['inttype', 'longtype', 'floattype', 'doubletype']
	file_datat = []
	f = code_file.replace(idg_file, '')
	f = code_file.replace(' ', '')
	f = f.split('_')
	f = [x.lower() for x in f]
	for i in f:
		if i in datat:
			file_datat.append(i)
	if len(dataTypes) == 0:
		sys.exit('ERROR, please select at least 1 data type')
	else:
		if (len(file_datat) == 0):
			if 'int' in dataTypes:
				return True
		else:
			for t in dataTypes:
				if t == 'all':
					isKeeper = True
				elif '~' in t: # exclude this type
					t = t.replace('~', '')
					if t in file_datat:
						return False
				elif 'only' in t: # need this type
					t = t.replace('only_', '')
					if t not in file_datat:
						return False
					else:
						isKeeper = True
				else: # type included
					t = t.replace('all_', '')
					if t in file_datat:
						isKeeper = True
	return isKeeper

def numBugs(f):
	num = 0
	for i in f:
		if 'bug' in i and 'no' not in i:
			num += 1
	return num

# filter options
def filter_opt(idg_file, code_file, options):
	isKeeper = False
	f = code_file.replace('_'.join(idg_file.split('_')[:2]), '')
	f = f.replace('_V_', '_Vertex_')
	f = f.replace('_E_', '_Edge_')
	f = f.split('_')
	if '' in f:
		f.remove('')
	f = [x.lower() for x in f]
	datat = ['inttype', 'longtype', 'floattype', 'doubletype']
	for i in datat:
		if i in f:
			f.remove(i)
	else:
		for b in options:
			if b == 'all': # all tags included by default unless otherwise excluded
				isKeeper = True
			elif '~' in b: # exclude this tag
				b = b.replace('~', '')
				if b in f:
					return False
			elif 'only' in b: # need this tag
				b = b.replace('only_', '')
				if b not in f:
					return False
				else:
					isKeeper = True
			else: # tag included
				b = b.replace('all_', '') # just in case
				if b in f:
					isKeeper = True
	return isKeeper

# read configure file
def read_configure(args):
	input_file = ''
	if os.path.isfile(config_defaultpath):
		input_file = open(config_defaultpath, 'r')
	elif len(args) >= 5 and os.path.isfile(args[4]):
		input_file = open(args[4], 'r')
	else:
		print("ERROR: configure.txt not found, specify location in arguments:", file=sys.stderr)
		sys.exit(error_msg)
	lines = input_file.readlines()
	codek = ['bug:', 'pattern:', 'option:', 'dataType:']
	num_codek = 4
	code_c = [[] for i in range(num_codek)]

	# read the code configure
	readc = False
	for l in lines:
		if '**/' in l:
			readc = True
		elif readc:
			for r in ((' ', ''), ('{', ''), ('}', ''), ('\t', ''), ('\n', '')):
				l = l.replace(*r)
			for k in codek:
				if k in l:
					idx = codek.index(k)
					l = l.replace(k, '')
					for i in l.split(','):
						code_c[idx].append(i.lower())
		if readc and ('INPUTS' in l):
			break;
	return code_c

def find_code(line_tags, count, p_all_tags1, re_split):
	# print(line_tags)
	# print(count)
	# print(p_all_tags1)
	# print(re_split)
	# print("end\n")
	code_result = ''
	idx = -1
	for j in range(0, len(line_tags)):
		for k in range(0, count):
			if line_tags[j] == p_all_tags1[k]:
				code_result = re_split[j + 1].strip()
				idx = j
				# print(code_result)
				# print(idx)
				# print("end1\n\n")
				return code_result, idx

def find_line_tags(l):
	re_tags = re.findall('\/\*\@[+]*[-]*[a-zA-Z]*[0-9]*\@\*\/', l)
	re_tags = [substr.replace('/*@', '@') for substr in re_tags]
	re_tags = [substr.replace('@*/', '@') for substr in re_tags]
	line_tags = [substr.replace('@', '') for substr in re_tags]
	return line_tags

# read the command line
args = sys.argv
if (len(args) <= 3):
	sys.exit(error_msg)

# change the save path
cur_path = os.getcwd()
out_path = cur_path
in_file_path = args[1]
out_directory = args[2]
model = -1
if args[3].lower() in model_map:
	model = model_map[args[3].lower()]
else:
	print("ERROR: Invalid programming_model argument, specify one of the following: C, CPP, OMP, CUDA\n", file=sys.stderr)
	sys.exit(error_msg)
save_path = os.path.join(out_path, out_directory)
file_name = (os.path.split(in_file_path))[1].replace('.idg', '')

if not os.path.isfile(in_file_path):
	sys.exit("ERROR: Invalid input_file argument")

# read code configuration
code_c = read_configure(args)

# filter pattern
f_pattern = False
pattern_c = code_c[1]
if len(pattern_c) == 0:
	sys.exit('ERROR, please select at least 1 pattern\n')
elif 'all' in pattern_c:
	f_pattern = True
for p in pattern_c:
	if '~' in p:
		p = p.replace('~', '')
		if p in file_name.lower():
			f_pattern = False
			break
	elif p in file_name.lower():
		f_pattern = True

num_codes = 0
num_bugs = 0
if (f_pattern):
	all_tags = [] # a list of tags per line
	nested_tags = [] # the nested tags that
	input_file = open(in_file_path, 'r')
	lines = input_file.readlines()

	add_nested = False

	# read code and tags
	for l in lines:
		if (re.search('\/\*\@[a-zA-Z]*[0-9]*\@\*\/', l)): # search the tag
			re_tags = re.findall('\/\*\@[a-zA-Z]*[0-9]*\@\*\/', l)
			re_tags = [substr.replace('/*@', '@') for substr in re_tags]
			re_tags = [substr.replace('@*/', '@') for substr in re_tags]
			line_tags = [substr.replace('@', '') for substr in re_tags]
			first_tag = line_tags[0]
			new_tag = True

			if ('+' not in first_tag) and ('-' not in first_tag):
				for i in all_tags:
					if i[0] == first_tag:
						new_tag = False
						break
				if new_tag:

					all_tags.append(list(line_tags))

	# # reorder the tags
	# tmp_tags = reorder_tags(tag_names, all_tags)
	# all_tags = tmp_tags

	# generate all combinations
	count = len(all_tags)
	p_all_tags = list(itertools.product(*all_tags))

	# keywords
	sup = 'suppress'
	blkLn = '//BlankLine'
	blkLn_nested = '//+BlankLine'
	dec = 'declare'

	# write to output folder
	for i in range(0, len(p_all_tags)):
		out_file_name = file_name
		# append the tags to the end of file name
		for j in range(0, len(p_all_tags[i])):
			if p_all_tags[i][j]:
				out_file_name = out_file_name + '_' + p_all_tags[i][j].strip()
		buggy = ((len(re.findall('No[a-zA-Z]*Bug[1-9]*', out_file_name)) != len(re.findall('[a-zA-Z]*Bug', out_file_name))))

		# filter bug, option, and data type
		ge_all = False
		f_bug = False
		bug_c = code_c[0]
		num_bugc = len(bug_c)
		if (num_bugc == 0) or (num_bugc > 2):
			sys.exit('ERROR, number bug configuration options must be smaller than 3.\n')
		else:
			if (num_bugc == 1):
				if 'hasbug' in bug_c:
					f_bug = True
				elif 'all' in bug_c:
					ge_all = True
				elif 'nobug' not in bug_c:
					sys.exit('ERROR, wrong configuration in bug.\n')
			else:
				ge_all = True
		f_type = filter_opt(file_name, out_file_name, code_c[2]) and filter_datat(file_name, out_file_name, code_c[3])

		# generate code
		if (ge_all or (f_bug == buggy)) and f_type:
			if not os.path.exists(save_path):
				os.chdir(out_path)
				os.mkdir(save_path)
				os.chdir(cur_path)
			out_file_name = out_file_name + model
			complete_file_name = os.path.join(save_path, out_file_name)
			output_file = open(complete_file_name, 'w')
			lspace = 0

			################ write copyright file
			if os.path.isfile(copyright_defaultpath):
				copyright_file = open(copyright_defaultpath, 'r')
			elif len(args) >= 6 and os.path.isfile(args[5]):
				copyright_file = open(args[5], 'r')
			else:
				print("ERROR: copyright.txt not found, specify location in arguments:", file=sys.stderr)
				sys.exit(error_msg)
			c_lines = copyright_file.readlines()
			output_file.write('/*\n')
			for l in c_lines: # write copyright
				output_file.write(l)
			output_file.write('\n*/\n\n\n')
			copyright_file.close()

			find_nested_tag = False
			nested = False
			flag = False
			act_tags = set() # activated tags
			prevTags = []

			for l in lines: # write codes
				l = l.strip()
				if (l == blkLn) or (find_nested_tag and l == blkLn_nested):
					output_file.write('\n')
					continue

				ostr = ''
				line_tags = find_line_tags(l)
				if nested: # in the nested tag section
					if line_tags and ('-' in line_tags[0]): # end the nested tag section
						nested = False
						find_nested_tag = False
						continue
					elif line_tags and find_nested_tag: # the +tag is activated
						re_split = re.split('\/\*\@[a-zA-Z]*[0-9]*\@\*\/', l)
						ostr, idx = find_code(line_tags, count, p_all_tags[i], re_split)
						
						if needBugComment(line_tags[idx], ostr):	# check if this nested line needs its own bug label
							ostr = "// " + line_tags[idx] + " here\n" + ' ' * lspace + ostr
						
						if sup in ostr:
							flag = True
							break
						elif dec not in ostr:
							act_tags.add(line_tags[idx])
					elif find_nested_tag:
						ostr = l.strip()
					
					if '+' in ' '.join(prevTags):	# if this is the first line of a bug tag block and isn't blank, add the bug label
						for ptag in prevTags:
							if needBugComment(ptag, ostr):	
								ostr = "// " + ptag.strip().replace('+','') + " here\n" + ' ' * lspace + ostr

				elif line_tags: # if the line has tags and not in the nested tag section
					re_split = re.split('\/\*\@[a-zA-Z]*[0-9]*\@\*\/', l)
					code = ''

					if ((not nested) and ('+' not in line_tags[0]) and ('-' not in line_tags[0])):
						re_split = re.split('\/\*\@[a-zA-Z]*[0-9]*\@\*\/', l)
						code, idx = find_code(line_tags, count, p_all_tags[i], re_split)
						
						if needBugComment(line_tags[idx], code):
							code = "// " + line_tags[idx] + " here\n" + ' ' * lspace + code
						
						if sup in code:
							flag = True
							break
						elif dec not in code:
							act_tags.add(line_tags[idx])
					elif (not nested) and ('+' in line_tags[0]):
						nested = True
						tcount = 0
						tmp = []
						or_tag = False
						for j in range(0, len(line_tags)):
							nested_tag_symbol = line_tags[j].replace('+', '')
							if nested_tag_symbol == 'or':
								or_tag = True
							else:
								for item in p_all_tags[i]:
									if item == nested_tag_symbol:
										tmp.append(item)
										tcount += 1

						if ((tcount == len(line_tags) and not or_tag)) or (or_tag and tcount >= 1):
							find_nested_tag = True
							for item in tmp:
								act_tags.add(item)
					if code:
						ostr = code
					prevTag = line_tags
				else:
					ostr = l
				if line_tags:
					prevTags = line_tags
				else:
					prevTags = []
				if ostr.startswith('}'):
					lspace = lspace - 2
				if ostr and dec not in ostr:
					output_file.write(' ' * lspace + ostr + '\n')
				if ostr.endswith('{'):
					lspace = lspace + 2
			output_file.close()
			# print(act_tags)
			# print(p_all_tags[i])

			if ((len(act_tags) != len(p_all_tags[i]))) or flag:
				os.remove(complete_file_name)
				#print("remove\n")
			else:
				num_codes += 1
				if buggy:
					num_bugs += 1
					#print("buggy\n")
					#print(act_tags)
	input_file.close()
print('Generating: %s' % file_name)
print('Number of codes generated: %d\n' % num_codes)
print('Number of buggy codes generated: %d\n' % num_bugs)