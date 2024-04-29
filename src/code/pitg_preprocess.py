from tokenizer import *
from util import *
from edit_path import *
from multiprocessing import Pool
import re
import tqdm
import json
import random

char_bag = string.ascii_letters + string.digits + string.punctuation
char_bag = set(char_bag)

region_bag = set(RegionTokenizer().bag)

SIM_THRESOLD = 0.3

stats = ["alpha", "gamma", "C1", "C2", "L1", "L2", "L3", "L4", "L5", "SM", "R1", "R2"]
se_list = ["C1", "C2", "L1", "L2", "L3", "L4", "L5", "SM", "R1", "R2"]

stats_map = {k: i for i, k in enumerate(stats)}

def tag_to_str(tag):
    return "".join([str(x) for x in tag])

def str_to_tag(str):
    return (str[0], int(str[1]))

def template_to_str(template):
    return "".join([tag_to_str(x) for x in template])

def generate_template_string(string):
    template = ""
    for char in string:
        if char.isalpha():
            template += 'L'
        elif char.isdigit():
            template += 'D'
        else:
            template += 'S'
    return template

def merge_and_count(pwd, string):
    if not string:
        return ()

    merged = []
    count = 1
    s = pwd[0]
    for i in range(1, len(string)):
        if string[i] == string[i-1]:
            count += 1
            s += pwd[i]
        else:
            merged.append(((string[i-1], count), s))
            count = 1
            s = pwd[i]
    # 处理最后一个字符
    merged.append(((string[-1], count), s))
    # List of (tag, segment)
    return tuple(merged)

def pcfg_decompose(x):
    template = generate_template_string(x)
    return merge_and_count(x, template)

def pwd_pcfg_movement(x):
    items = pcfg_decompose(x)
    segments = [x[1] for x in items]
    segments = segments[1:] + [segments[0]]
    return "".join(segments)

def pwd_reverse(x):
    return x[::-1]

def pwd_segment_reverse(x):
    items = pcfg_decompose(x)
    segments = [x[1][::-1] for x in items]
    return "".join(segments)

TRANSFORMER_FN = {
    "No": lambda x:x, 
    "C1" : lambda x:x.upper(), 
    "C2" : lambda x:x.lower(), 
    "L1": lambda x: x.replace("a", "@"), 
    "L2": lambda x: x.replace("s", "$"), 
    "L3": lambda x: x.replace("o", "0"),
    "L4": lambda x: x.replace("i", "1"),
    "L5": lambda x: x.replace("e", "3"), 
    "SM": pwd_pcfg_movement, 
    "R1": pwd_reverse, 
    "R2": pwd_segment_reverse
}

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

def sim(pwd1, pwd2, threshold=SIM_THRESOLD):
    return cosine_similarity(pwd1, pwd2) > threshold

def one_hot_cnt(key):
    ans = [0] * len(stats)
    if key in stats_map:
        ans[stats_map[key]] += 1
    return ans

class PITGCounter:
    def __init__(self):
        self.map = defaultdict(lambda :defaultdict(int))
        self.region_cnt = defaultdict(int)
        self.total = 0
        self.valid = 0
    
    def add(self, ri, k):
        self.total += 1
        if k != None:
            self.valid += 1
            self.region_cnt[ri] += 1
            self.map[ri][k] += 1
    
    def normalize(self):
        keys = self.region_cnt.keys()
        for ri in keys:
            for k in stats:
                self.map[ri][k] = self.map[ri][k] / self.region_cnt[ri]
    
    def save(self, path):
        self.normalize()
        with open(path, "w") as f:
            json.dump({
                "weights": self.map, 
                "region_cnt": self.region_cnt
            }, f, indent="  ")
        print(f">>> Weights saved in {path}")
        print(f">>> Total: {self.total}")
        print(f">>> Valid: {self.valid}, ({self.valid / self.total})")

def record_counter(item):
    pwd1, pwd2, email = item
    ui, hi, ri = parse_email(email)
    if ri not in region_bag:
        ri = ".com"
    if sim(ui, pwd2, threshold=0.6):
        return ri, "gamma"
    if sim(pwd1, pwd2):
        return ri, "alpha"
    for k in se_list:
        if TRANSFORMER_FN[k](pwd1) == pwd2:
            return ri, k
    return ri, None

def _run_parallel(chunks, cpu, chunk_size):
    ans = []
    with Pool(cpu) as p:
        ret = p.map(record_counter, chunks, chunksize=chunk_size)
        ans.extend(ret)
    return ans

def random_index(pwds):
    n = len(pwds)
    if n <= 1:
        raise Exception("At least two passwords")
    

def parse_line(line):
    line = line.split("\t")
    if len(line) != 2:
        raise Exception("Not valid line")
    email = line[0]
    pwds = json.loads(line[1][1:-1])
    pwd1, pwd2 = random.sample(pwds, 2)
    return pwd1, pwd2, email
    
def parallel_apply(csv_path, output_path, cpu=10, chunk_size=10000):
    error_cnt = 0
    total = 0
    items = []
    with open(csv_path, "r") as f2:
        for line in f2:
                line = line.strip("\r\n")
                try:
                    pwd1, pwd2, email = parse_line(line)
                except Exception as e:
                    continue
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
    progress_bar = tqdm.tqdm(total=total)
    weights = PITGCounter()
    while True:
        csvdata = items[idx:min(idx+chunk_size*cpu, total)]
        if len(csvdata) <= 0:
            break
        idx += len(csvdata)
        values = _run_parallel(csvdata, cpu, chunk_size)
        for item in values:
            weights.add(*item)
        progress_bar.update(len(csvdata))
    progress_bar.close()

    weights.save(output_path)

def main():
    data_path = "/disk/data/targuess/0_sample/Collection1_4kw.csv"
    weight_save = "/disk/yjt/PersonalTarGuess/model/pitg_weights/Collection1_cos_4kw.json"
    parallel_apply(data_path, weight_save, cpu=10, chunk_size=10000)
    pass

if __name__ == '__main__':
    main()