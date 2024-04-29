from collections import defaultdict
import string
import math
import tqdm

def valid_pwd(pwd):
    valid_characters = string.ascii_letters + string.digits + string.punctuation
    return all(char in valid_characters for char in pwd)

def read_pwd_list(file_path):
    frequency = defaultdict(int)
    with open(file_path, 'r') as file:
        for line in file:
            string = line.strip()
            if valid_pwd(string):
                frequency[string] += 1
    return frequency

def count_lines(filename):
    num_lines = 0
    with open(filename, 'r') as file:
        for _ in file:
            num_lines += 1
    return num_lines

def read_pwd_pairs(file_path, with_bar=False):
    if with_bar:
        cnt = count_lines(file_path)
    else:
        cnt = 0
    with open(file_path, 'r') as file:
        if with_bar:
            file = tqdm.tqdm(file, desc="Parse Pairs", unit="lines", total=cnt)
        for line in file:
            first, second = line.strip().split('\t')
            if valid_pwd(first) and valid_pwd(second):
                yield first, second

def normalize_dict(D):
    total = sum([v for v in D.values()])
    if total == 0:
        return D
    keys = [k for k in D.keys()]
    ans = []
    for k in keys:
        ans.append((k, D[k] / total))
    return sorted(ans, key=lambda x:x[1], reverse=True)

def combine_segment(segment):
    return "".join(segment)

def log(x):
    if x > 0:
        return -math.log2(x)
    return 50

def main():
    pass


if __name__ == '__main__':
    main()