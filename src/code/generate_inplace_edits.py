# We apply edit distance to password similarity detect.
# For example, the source password is intention and the target password is execution
# Then we first get:
# inte*ntion
# *execution
# dsskiskkkk
# *nt*iu****
#
# The second step, eliminate "insert" by x(replace current letter by two letters)
# 
# inte*ntion
# *execution
# dssxkskkkk
# *ntecu****

import itertools
import json
import string
import numpy as np
import argparse
from multiprocessing import Queue, Process, Lock
import logging
import datetime
from multiprocessing import Pool
import tqdm

char_bag = string.ascii_letters + string.digits + string.punctuation
char_bag = set(char_bag)

def load_password_letter():
    LETTERS = string.ascii_letters
    NUMBERS = string.digits
    SPECIALS = string.punctuation
    SPACE = " "
    return LETTERS + NUMBERS + SPECIALS + SPACE

def load_inplace_trans_dict():
    actions = {}
    # load simple operations
    actions[('k',None)] = len(actions)
    actions[('d',None)] = len(actions)
    letters = load_password_letter()
    for ch in letters:
        actions[('s',ch)] = len(actions)
    # load complex operations
    for ch in letters:
        for ch2 in letters:
            actions[('x',ch+ch2)] = len(actions)
    return actions

def load_reverse_trans_dict():
    mapper = load_inplace_trans_dict()
    ans = {}
    for i,item in mapper.items():
        ans[item] = i
    return ans

mapper = load_inplace_trans_dict()

def find_med_backtrace(str1, str2, cutoff=-1):
    '''
    This function calculates the Minimum Edit Distance between 2 words using
    Dynamic Programming, and asserts the optimal transition path using backtracing.
    Input parameters: original word, target word
    Output: minimum edit distance, path
    Example: ('password', 'Passw0rd') -> 2.0, [('s', 'P', 0), ('s', '0', 5)]
    '''
    # op_arr_str = ["d", "i", "c", "s"]

    # Definitions:
    n = len(str1)
    m = len(str2)
    D = np.full((n + 1, m + 1), np.inf)
    trace = np.full((n + 1, m + 1), None)
    trace[1:, 0] = list(zip(range(n), np.zeros(n, dtype=int)))
    trace[0, 1:] = list(zip(np.zeros(m, dtype=int), range(m)))
    # Initialization:
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)

    # Fill the matrices:
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            delete = D[i - 1, j] + 1
            insert = D[i, j - 1] + 1
            if (str1[i - 1] == str2[j - 1]):
                sub = np.inf
                copy = D[i - 1, j - 1]
            else:
                sub = D[i - 1, j - 1] + 1
                copy = np.inf
            op_arr = [delete, insert, copy, sub]
            D[i, j] = np.min(op_arr)
            op = np.argmin(op_arr)
            if (op == 0):
                # delete, go down
                trace[i, j] = (i - 1, j)
            elif (op == 1):
                # insert, go left
                trace[i, j] = (i, j - 1)
            else:
                # copy or subsitute, go diag
                trace[i, j] = (i - 1, j - 1)
    # print(trace)
    # Find the path of transitions:
    i = n
    j = m
    cursor = trace[i, j]
    path = []
    while (cursor is not None):
        # 3 possible directions:
        #         print(cursor)
        if (cursor[0] == i - 1 and cursor[1] == j - 1):
            # diagonal - sub or copy
            if (str1[cursor[0]] != str2[cursor[1]]):
                # substitute
                path.append(("s", str2[cursor[1]], cursor[0]))
            i = i - 1
            j = j - 1
        elif (cursor[0] == i and cursor[1] == j - 1):
            # go left - insert
            path.append(("i", str2[cursor[1]], cursor[0]))
            j = j - 1
        else:
            # (cursor[0] == i - 1 and cursor[1] == j )
            # go down - delete
            path.append(("d", None, cursor[0]))
            i = i - 1
        cursor = trace[cursor[0], cursor[1]]
        # print(len(path), cursor)
    md = D[n, m]
    del D, trace
    return md, list(reversed(path))



# Decoder - given a word and a path of transition, recover the final word:
def path2word(word, path):
    '''This function decodes the word in which the given path transitions the input
    word into.  Input parameters: original word, transition path Output: decoded
    word

    '''
    if not path:
        return word
    final_word = []
    word_len = len(word)
    path_len = len(path)
    i = 0
    j = 0
    while (i < word_len or j < path_len):
        if (j < path_len and path[j][2] == i):
            if (path[j][0] == "s"):
                # substitute
                final_word.append(path[j][1])
                i += 1
                j += 1
            elif (path[j][0] == "d"):
                # delete
                i += 1
                j += 1
            else:
                # "i", insert
                final_word.append(path[j][1])
                j += 1
        else:
            final_word.append(word[i])
            i += 1
    return ''.join(final_word)

def inplace_mask(pwd, edits):
    # inplace_op = [['k',None] for _ in range(len(pwd1))]
    mask = [0 for ch in pwd]
    for op in edits:
        if op[0] == 's' or op[0] == 'd':
            mask[op[2]] = 1
    for op in edits:
        if op[0] != 'i':
            continue
        pos = op[2]
        char = op[1]
        if pos >= len(pwd):
            mask.append(1)
            continue
        mask[pos] = 1
    return mask

def inplace_edit(pwd1, pwd2):
    path = find_med_backtrace(pwd1, pwd2)
    dist = path[0]
    edits = path[1]
    is_arrive = True
    # inplace_op = [['k',None] for _ in range(len(pwd1))]
    inplace_op = [['s',ch] for ch in pwd1]
    for op in edits:
        if op[0] == 's' or op[0] == 'd':
            inplace_op[op[2]] = [op[0], op[1]]
    for op in edits:
        if op[0] != 'i':
            continue
        pos = op[2]
        char = op[1]
        if pos >= len(pwd1):
            inplace_op.append(['s',char])
            continue
        if inplace_op[pos][0] == 'k':
            inplace_op[pos] = ['x',char+pwd1[pos]]
        elif inplace_op[pos][0] == 's':
            inplace_op[pos] = ['x',char+inplace_op[pos][1]]
        else:
            is_arrive = False
    if len(inplace_op)- len(pwd1) > 3:
        is_arrive = False
    return dist, inplace_op if is_arrive else [], is_arrive

def valid_pwd(pwd, min_len=5, max_len=16, ban_chars=['"', ',']):
    if len(pwd) < min_len or len(pwd) > max_len:
        return False
    for char in ban_chars:
        if char in pwd:
            return False
    return all([x in char_bag for x in pwd])

def pair_reader(csv_file):
    for line in open(csv_file, "r"):
        line = line.strip().split(",",1)
        if len(line[1]) <= 4:
            continue
        if len(line) == 2:
            pwd_list = []
            try:
                pwd_list = json.loads(line[1])
            except json.decoder.JSONDecodeError:
                print(line)
                continue
            yield (line[0], pwd_list)

def encode_inplace_edit(path):
    res = [0 for _ in range(len(path))]
    for i, op in enumerate(path):
        if(tuple(op) not in mapper):
            op = ['d',None]
        res[i] = mapper[tuple(op)]
    return res

def record2path(item):
    pwd1, pwd2 = item 
    dist, edits, is_arrive = inplace_edit(pwd1, pwd2)
    if is_arrive:
        edits = encode_inplace_edit(edits)
    return (pwd1, pwd2, edits, is_arrive)

def recover_inplace_edit(pwd, decode_path):
    res = [ch for ch in pwd] + ['', '', '']
    for i, item in enumerate(decode_path):
        if item[0] == 'k':
            continue
        if item[0] == 's' or item[0] == 'x':
            res[i] = item[1]
        if item[0] == 'd':
            res[i] = ''
    return "".join(res).strip(' ')

def _run_parallel(chunks, cpu, chunk_size):
    ans = []
    with Pool(cpu) as p:
        ret = p.map(record2path, chunks, chunksize=chunk_size)
        ans.extend(ret)
    return ans

def parallel_apply(csv_path, output_path, cpu=10, chunk_size=10000):
    error_cnt = 0
    total = 0
    items = []
    with open(csv_path, "r") as f2:
        for line in f2:
                line = line.strip("\r\n")
                line = line.split("\t")
                if len(line) != 3:
                    error_cnt += 1
                    continue
                pwd1 = line[0]
                pwd2 = line[1]
                email = line[2]
                if not valid_pwd(pwd1) or not valid_pwd(pwd2):
                    error_cnt += 1
                    continue 
                items.append((pwd1, pwd2))
                total += 1
    print(f">>> Load data from :{csv_path}")
    print(f">>> Total cnt: {total}")
    print(f">>> Error cnt: {error_cnt}")
    print(f">>> Processing")
    idx = 0
    cnt = 0
    progress_bar = tqdm.tqdm(total=total)
    with open(output_path, "w") as f1:
        while True:
            csvdata = items[idx:min(idx+chunk_size*cpu, total)]
            if len(csvdata) <= 0:
                break
            idx += len(csvdata)
            values = _run_parallel(csvdata, cpu, chunk_size)
            for item in values:
                # print(item)
                if len(item) != 4:
                    continue
                if not item[3]:
                    continue 
                f1.write(f"{item[0]}\t{item[1]}\t{item[2]}\n")
                cnt += 1
            progress_bar.update(len(csvdata))
    progress_bar.close()
    print(f">>> Write cnt: {cnt}")

def main():
    data_path = "/disk/data/targuess/1_triplet/Collection1_cos_4kw.csv"
    save_path = "/disk/data/targuess/2_train/passbert/Collection1_cos_4kw.csv"
    # read_triplet(data_path, save_path)
    print(f">>> Size of Trans: {len(mapper)}")
    parallel_apply(data_path, save_path, cpu=10)
    print(f">>> Result saved in : {save_path}")

if __name__ == '__main__':
    main()