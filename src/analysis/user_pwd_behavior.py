import itertools
import json
import string
import numpy as np
import argparse
from multiprocessing import Queue, Process, Lock
import logging
import datetime
from collections import defaultdict
import random
import re
from collections import Counter
import tqdm

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

def pair_reader(csv_file):
    for line in open(csv_file, "r"):
        line = line.strip().split("\t",1)
        if len(line[1]) <= 4:
            continue
        if len(line) == 2:
            pwd_list = []
            try:
                pwd_list = json.loads(line[1][1:-1])
            except json.decoder.JSONDecodeError:
                print(line)
                continue
            yield (line[0], pwd_list)

def parse_email(email):
    pattern = r'^([^@]+)@([^@]+)(\.[^@]+)$'
    matches = re.match(pattern, email)
    if matches:
        username = matches.group(1)
        domain = matches.group(2)
        top_level_domain = matches.group(3)
        return username, domain.lower(), top_level_domain.lower()
    else:
        return "unk", "unk", "unk"

def get_topk_elements(dictionary, k):
    counter = Counter(dictionary)
    return counter.most_common(k)

def top_values(cnt, topk=100):
    items = get_topk_elements(cnt, topk)
    items = {str(x[0]): str(x[1]) for x in items}
    return items

def top_item_values(cnt, counter, topk=100, topv=10):
    items = get_topk_elements(cnt, topk)
    items = [x[0] for x in items]
    res = {}
    for key in items:
        values = get_topk_elements(counter[key], topv)
        res[key] = {str(k): str(v) for k,v in values}
    return res

def init_cli():
    cli = argparse.ArgumentParser("User edit behavior analysis")
    cli.add_argument("-i", dest="input")
    cli.add_argument("-o", dest="save")
    cli.add_argument("-n", dest="cnt", default=1000000, type=int)
    args = cli.parse_args()
    return {
        "input": args.input, 
        "save": args.save, 
        "cnt": args.cnt
    }

def main():

    args = init_cli()

    data_path = args["input"]

    result_save = args["save"]

    total_cnt = defaultdict(int)
    region_cnt = defaultdict(lambda :defaultdict(int))
    host_cnt = defaultdict(lambda :defaultdict(int))
    cnt1 = defaultdict(int)
    cnt2 = defaultdict(int)

    for email, data in tqdm.tqdm(pair_reader(data_path), total=args["cnt"]):
        username, hostname, region = parse_email(email)
        if len(data) > 5:
            data = data[:5]
        min_d = 100
        for pwd in data:
            d, dist = find_med_backtrace(username, pwd)
            min_d = min(min_d, d)
        min_d = int(min_d)
        key = str(min_d)
        region_cnt[region][key] += 1
        host_cnt[hostname][key] += 1
        total_cnt[key] += 1
        cnt1[region] += 1
        cnt2[hostname] += 1
    
    res = {
        "total_cnt": total_cnt, 
        "region_cnt": top_item_values(cnt1, region_cnt, 100, None), 
        "host_cnt": top_item_values(cnt2, host_cnt, 100, None)
    }

    with open(result_save, "w") as f:
        json.dump(res, f, indent="\t")
    
    print(f">>> Result saved in {result_save}")


main()
