
"""
Author: Tal Daniel
Minimum Edit Distance with Backtrace
-----------------------------------

We wish to find the Minimum Edit Distance (MED) between two strings. That is,
given two strings, align them, and find the minimum operations from {Insert,
Delete, Substitute} needed to get from the first string to the second string.
Then, we want to find the actual operations done in order to reach this MED,
e.g "Insert 'A' at position 3".

We can try and achieve this goal using Dynamic Programming (DP) for optimal
complexity as follows: Define:
* String 1: $X$ of length $n$
* String 2: $Y$ of length $m$
* $D[i,j]$: Edit Distance between substrings $X[1 \rightarrow i]$ and $Y[1 \rightarrow j]$

Using "Bottom Up" approach, the MED between $X$ and $Y$ would be $D[n,m]$.

We assume that the distance between string of length 0 to a string of length k
is k, since we need to insert k characters is order to create string 2.  In
order to actually find the operation, we need to keep track of the operations,
that is, create a "Backtrace".

"""

import random
import numpy as np
import string
import json
import csv
import itertools
import time
from ast import literal_eval
from pathlib import Path
from tokenizer import TransTokenizer
import string 
import re
from multiprocessing import Pool
import tqdm

char_bag = string.ascii_letters + string.digits + string.punctuation
char_bag = set(char_bag)

TRANS_TOKENIZER = TransTokenizer()
TRANS_to_IDX = TRANS_TOKENIZER.get_encode_dict()
IDX_to_TRANS = TRANS_TOKENIZER.get_decode_dict()

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

def path2idx(path):
    idx_path = [TRANS_to_IDX.get(p, -1) for p in path]
    return idx_path

def idx2path(path):
    str_path = [IDX_to_TRANS.get(str(p), ("<unk>", "<unk>", -1))
                for p in path]
    return str_path

def csv2pws_pairs_gen(csv_fpath, line_s=0, line_e=None):
    with open(csv_fpath) as csv_file:
        for i, row in enumerate(csv_file):
            row = row.strip('\r\n')
            row = row.split(",",1)
            if (len(row) != 2):
                print("File format error @ line {}\n{!r}!".format(i, row))
                break
            username, pws_string = row
            # pws_list = eval(pws_string) # In case arrays are not json
            # formatted
            try:
                # print(pws_string)
                pws_list = json.loads(pws_string)
                
            except json.decoder.JSONDecodeError as ex:
                print(ex)
                continue
            for p in itertools.permutations(pws_list, 2):
                yield p

def pair2path(kw1, kw2):
    med, path = find_med_backtrace(kw1, kw2)
    path_indices = path2idx(path)
    print(path, path_indices)
    # for testing
    if random.randint(0, 1000) <= 10:
        decoded_word = path2word(kw1, path)
        if (decoded_word != kw2):
            print("Test failed on: {}".format((kw1, kw2)))
            print("Path chosen: {}".format(path))
            print("Decoded Password: {}".format(decoded_word))
    return path_indices

def record2path(record):
    kw1, kw2, email = record
    med, path = find_med_backtrace(kw1, kw2)
    
    path_indices = path2idx(path)
    # for testing
    if random.randint(0, 1000) <= 10:
        decoded_word = path2word(kw1, path)
        if (decoded_word != kw2):
            print("Test failed on: {}".format((kw1, kw2)))
            print("Path chosen: {}".format(path))
            print("Decoded Password: {}".format(decoded_word))
    return (kw1, kw2, email, path_indices)

def valid_pwd(pwd, min_len=5, max_len=16, ban_chars=['"', ',']):
    if len(pwd) < min_len or len(pwd) > max_len:
        return False
    for char in ban_chars:
        if char in pwd:
            return False
    return all([x in char_bag for x in pwd])

def valid_email(email):
    pattern = r'^([^@]+)@([^@]+)(\.[^@]+)$'
    matches = re.match(pattern, email)
    if matches:
        return True
    else:
        return False

def read_triplet(csv_path, output_path):
    error_cnt = 0
    total = 0
    with open(output_path, "w") as f1:
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
                path = pair2path(pwd1, pwd2)
                f1.write(f"{pwd1}\t{pwd2}\t{email}\t{json.dumps(path)}\n")
                total += 1
    print(f">>> Load data from :{csv_path}")
    print(f">>> Total cnt: {total}")
    print(f">>> Error cnt: {error_cnt}")

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
                if not valid_email(email):
                    error_cnt += 1
                    continue
                items.append((pwd1, pwd2, email))
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
                f1.write(f"{item[0]}\t{item[1]}\t{item[2]}\t{item[3]}\n")
                cnt += 1
            progress_bar.update(len(csvdata))
    progress_bar.close()
    print(f">>> Write cnt: {cnt}")

def main():
    
    data_path = "/disk/data/targuess/1_triplet/Collection1_cos_4kw.csv"
    save_path = "/disk/data/targuess/2_train/pitg/Collection1_cos_4kw.csv"
    # read_triplet(data_path, save_path)
    print(f">>> Size of Trans: {len(TRANS_TOKENIZER)}")
    parallel_apply(data_path, save_path, cpu=10)
    print(f">>> Result saved in : {save_path}")

    print(TRANS_to_IDX[('d', None, 6)])
    pass

if __name__ == '__main__':
    main()