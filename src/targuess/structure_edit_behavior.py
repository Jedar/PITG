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
from structure_transform import *

region_keys = [".com", ".ru", ".fr", ".uk", ".cn", ".jp"]

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

def random_index(max_id):
    value1 = int(random.random()*max_id)
    value2 = int(random.random()*max_id)
    while value1 == value2:
        value2 = int(random.random()*max_id)
    return (value1, value2)

def random_data(pwds):
    if len(pwds) < 2:
        return pwds
    n = len(pwds)
    i1, i2 = random_index(n)
    return pwds[i1], pwds[i2]

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

def top_values(cnt, counter, topk=100, topv=10):
    items = get_topk_elements(cnt, topk)
    items = [x[0] for x in items]
    res = {}
    for key in items:
        values = get_topk_elements(counter[key], topv)
        res[key] = [{str(k): str(v)} for k,v in values]
    return res

def main():

    data_path = "/disk/yjt/PersonalTarGuess/data/analysis/Collections1_500w.csv"

    counter1 = defaultdict(lambda :defaultdict(int))
    counter2 = defaultdict(lambda :defaultdict(int))
    counter3 = defaultdict(int)
    total = 0

    lines = count_lines(data_path)

    for email, data in tqdm.tqdm(pair_reader(data_path), total=lines):
        pwd1, pwd2 = random_data(data)
        username, hostname, region = parse_email(email)
        p =pwd1
        total += 1
        if p == pwd2:
            continue
        for k1 in TOTAL_TRANSFORMERS:
            counted = False
            for k2 in TRANSFORMERS[k1]:
                pwd = TRANSFORMER_FN[k2](p)
                if pwd == pwd2:
                    counter1[k1][k2] += 1
                    counted = True
            if counted:
                counter1[k1]["count"] += 1
                counter2[region][k1] += 1
        
        counter2[region]["count"] += 1
    
    for region in region_keys:
        print(f">>> Region: {region}")
        if counter2[region]["count"] == 0:
            continue
        cnt = counter2[region]["count"]
        for k in TOTAL_TRANSFORMERS:
            print(f">>> {k}: {counter2[region][k]}, {counter2[region][k]/cnt}")

    
    print(f">>> Total: {total}")
    for k1 in TOTAL_TRANSFORMERS:
        if counter1[k1]["count"] == 0:
            continue
        cnt = counter1[k1]["count"]
        for k2 in TRANSFORMERS[k1]:
            print(f">>> {k1} - {k2}, {counter1[k1][k2]}, {counter1[k1][k2]/total}")
        print(f">>> Ratio {k1}: {cnt/total}")

main()
