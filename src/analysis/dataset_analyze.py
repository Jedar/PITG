import json
from collections import defaultdict
from ast import literal_eval

path = "/disk/data/with_email/Collections1_reduce.csv"

cnt = set()

total = 0

MAX_LEN=16

max_cnt = 0

def parse_line(pwds):
    try:
        ans = literal_eval(pwds[1:-1])
    except:
        ans = []
    return ans

analysis_cnt = 50000000

with open(path, "r") as f:
    for line in f:
        line = line.strip("\r\n").split("\t")
        if len(line) <= 1:
            continue
        pwds = parse_line(line[1])
        total += len(pwds)
        for pwd in pwds:
            if len(pwd) > MAX_LEN:
                max_cnt += 1
        if total > analysis_cnt:
            break
            # cnt.add(pwd)

print(f">>> Total pwds: {total}")
print(f">>> Pwds exceed max len: {max_cnt}, {max_cnt / total}")
# print(f">>> Unique pwds: {len(cnt)}")