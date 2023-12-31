#!/usr/bin/python3 -u

import itertools
import sys
import os
import re

# filter data type
def filter_datat(idg_file, code_file, dataType):
	if 'all' in dataType:
		return True
	datat = ['IntType', 'short', 'LongType', 'FloatType', 'DoubleType', 'CharType']
	file_datat = []
	f = code_file.replace(idg_file, '')
	f = code_file.replace(' ', '')
	f = f.split('_')
	for i in f:
		if i in datat:
			file_datat.append(i)
	if len(dataType) == 0:
		sys.exit('ERROR, please select at least 1 data type')
	elif 'all' in dataType:
		return True
	else:
		if (len(file_datat) == 0):
			if 'int' in dataType:
				return True
		else:
			for d in file_datat:
				if ('~' not in dataType) and (d in dataType):
					return True
				elif ('~' in dataType) and (d not in dataType):
					return True
	return False

def numBugs(f):
	num = 0
	for i in f:
		if 'Bug' in i and 'No' not in i:
			num += 1
	return num

# filter options
def filter_opt(idg_file, code_file, options):
	if 'all' in options:
		return True
	f = code_file.replace(idg_file, '')
	f = f.split('_')
	f.remove('')
	datat = ['IntType', 'short', 'LongType', 'float', 'double', 'char']
	for i in datat:
		if i in f:
			f.remove(i)
	else:
		for b in options:
			if 'only' in b:
				b = b.replace('only_', '')
				if (b in f) and (numBugs(f) == 1):
					return True
			elif '~' in b:
				b = b.replace('~', '')
				if b not in f:
					return True
			else:
				b = b.replace('all_', '')
				if b in f:
					return True
	return False

# read configure file
def read_configure(cfile):
	input_file = open(cfile, 'r')
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
						code_c[idx].append(i)
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
	sys.exit('USAGE: input_file_name output_directory programming_model\n')

# change the save path
cur_path = os.getcwd()
out_path = cur_path
in_file_path = args[1]
out_directory = args[2]
model = "."
model += args[3]
save_path = os.path.join(out_path, out_directory)
file_name = (os.path.split(in_file_path))[1].replace('.idg', '')

# read code configuration
code_c = read_configure('codeGen/configure.txt')

# filter pattern
f_pattern = False
pattern_c = code_c[1]
if len(pattern_c) == 0:
	sys.exit('ERROR, please select at least 1 pattern\n')
elif 'all' in pattern_c:
	f_pattern = True
else:
	for p in pattern_c:
		if p in file_name:
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
			# copyright_file = open(args[2], 'r')
			# c_lines = copyright_file.readlines()
			# output_file.write('/* ')
			# for l in c_lines: # write copyright
			# 	output_file.write(l)
			# output_file.write(' */\n\n')
			# copyright_file.close()

			find_nested_tag = False
			nested = False
			flag = False
			act_tags = set() # activated tags

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

						if sup in ostr:
							flag = True
							break
						elif dec not in ostr:
							act_tags.add(line_tags[idx])
					elif find_nested_tag:
						ostr = l.strip()

				elif line_tags: # if the line has tags and not in the nested tag section
					re_split = re.split('\/\*\@[a-zA-Z]*[0-9]*\@\*\/', l)
					code = ''

					if ((not nested) and ('+' not in line_tags[0]) and ('-' not in line_tags[0])):
						re_split = re.split('\/\*\@[a-zA-Z]*[0-9]*\@\*\/', l)
						code, idx = find_code(line_tags, count, p_all_tags[i], re_split)
						if sup in code:
								#print("true")
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

						if ((tcount == len(line_tags) and not or_tag)) or (or_tag and tcount == 1):
							find_nested_tag = True
							for item in tmp:
								act_tags.add(item)
					if code:
						ostr = code
				else:
			 		ostr = l
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