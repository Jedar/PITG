from collections import defaultdict, Counter
import json
import heapq
import tqdm

from pcfg_basic import *
from util import *
from edit_path import *
from structure_transform import *

def build_default_transformer_map():
    total_trans = ["C", "L", "SM", "R"]
    c_trans = ["No", "C1", "C2", "C3", "C4"]
    l_trans = ["No", "L1", "L2", "L3", "L4", "L5"]
    sm_trans = ["No", "SM"]
    r_trans = ["No", "R1", "R2"]
    return {
        "C": {k:0 for k in c_trans}, 
        "L": {k:0 for k in l_trans}, 
        "SM": {k:0 for k in sm_trans},
        "R": {k:0 for k in r_trans}
    }

def build_default_edit_map():
    total_edits = ["No", "hi", "ti", "hd", "td"]
    return {
        "No": [0, defaultdict(int)], 
        "hi": [0, defaultdict(int)], 
        "ti": [0, defaultdict(int)], 
        "hd": [0, defaultdict(int)], 
        "td": [0, defaultdict(int)]
    }


class TarGuessII:
    def __init__(
            self, 
            popular_size=10000, 
            similar_threshold=0.5):
        self.popular_pwds = None
        self.segment_map = None
        self.transformer_map = None
        self.structure_edit_map = None
        self.segment_edit_map = None
        self.similar_threshold = similar_threshold
        self.popular_size = popular_size
        self.total_trained = 0
        self.popular_cnt = 0
        self.build_map()
        pass

    def build_map(self):
        self.segment_map = None
        self.popular_pwds = []
        self.transformer_map = defaultdict(lambda :defaultdict(int))
        self.structure_edit_map = defaultdict(lambda :{
            "No": [1], 
            "td": [1],
            "hd": [1],
            "ti": [1, defaultdict(int)],
            "hi": [1, defaultdict(int)]
        })
        self.segment_edit_map = defaultdict(lambda :{
            "No": [1], 
            "td": [1],
            "hd": [1],
            "ti": [1, defaultdict(int)],
            "hi": [1, defaultdict(int)]
        })
    
    @staticmethod
    def normalize_edit_map(map):
        keys = [k for k in map.keys()]
        for k in keys:
            total = sum([map[k][x][0] for x in map[k]])
            if total == 0:
                continue
            pp = [p for p in map[k].keys()]
            for x in pp:
                map[k][x][0] = map[k][x][0] / total
                if x in ["hi", "ti"]:
                    map[k][x][1] = normalize_dict(map[k][x][1])

    def normalize(self):
        keys = [k for k in self.transformer_map.keys()]
        for k in keys:
            self.transformer_map[k] = normalize_dict(self.transformer_map[k])
        
        self.normalize_edit_map(self.structure_edit_map)
        self.normalize_edit_map(self.segment_edit_map)
        pwds = []
        for k,v in self.popular_pwds.items():
            if self.popular_cnt > 0:
                pwds.append((k, log(v / self.popular_cnt)))
            else:
                pwds.append((k, 50))
        self.popular_cnt = self.popular_cnt / self.total_trained
        self.popular_pwds = sorted(pwds, key=lambda x:x[1])

    def train_popular(self, popular_path):
        pwds = read_pwd_list(popular_path)

        print(f">>> Load popular pwds: {popular_path}, Size: {len(pwds)}")

        counter = Counter({k:v for k,v in pwds.items() if len(k) >= 8})
        popular_pwds = heapq.nlargest(self.popular_size, counter.items(), key=lambda x:x[1])
        self.popular_pwds = {k[0]:0 for k in popular_pwds}

        self.segment_map = train_basic_pcfg(pwds)
    
    def train(self, pair_path):
        pair_cnt = count_lines(pair_path)
        print(f">>> Load pwd pairs: {pair_path}, Size: {pair_cnt}")

        pairs = read_pwd_pairs(pair_path, with_bar=True)

        for pwd1, pwd2 in pairs:
            self.train_pair(pwd1, pwd2)
        
        self.normalize()
    
    def count_transformer_map(self, cnt):
        for k in TOTAL_TRANSFORMERS:
            # print(">>> cnt: ", cnt)
            self.transformer_map[k][cnt[k]] += 1

    def train_pair(self, pwd1, pwd2):
        self.total_trained += 1
        if pwd2 in self.popular_pwds:
            self.popular_cnt += 1
            self.popular_pwds[pwd2] += 1
            return 
        valid, closest, cnt = get_closest_pwd(pwd1, pwd2, self.similar_threshold)
        if not valid:
            return 
        self.count_transformer_map(cnt)
        structure_edits, segment_edits = generate_pcfg_structure_edit(closest, pwd2)
        
        # print(structure_edits, segment_edits)

        for edit in structure_edits:
            self.structure_edit_map[edit[0]][edit[1]][0] += 1
            if edit[1] in ["hi", "ti"]:
                self.structure_edit_map[edit[0]][edit[1]][1][edit[2]] += 1
        
        for edit in segment_edits:
            self.segment_edit_map[edit[0]][edit[1]][0] += 1
            if edit[1] in ["hi", "ti"]:
                self.segment_edit_map[edit[0]][edit[1]][1][edit[2]] += 1
        pass

    def save(self, path):
        model = {
            "segment_map": self.segment_map, 
            "transformer_map": self.transformer_map, 
            "structure_edit_map": self.structure_edit_map,
            "segment_edit_map": self.segment_edit_map, 
            "popular_pwds": self.popular_pwds, 
            "popular_ratio": self.popular_cnt
        }
        with open(path, "w") as f:
            json.dump(model, f, indent="  ")
        print(f">>> Targuess II model saved in :{path}")
        pass

    def load(self, path):
        pass


def main():

    popular_pwds = "/disk/data/general/rockyou_new.txt"
    pwd_pairs = "/disk/data/targuess/2_train/targuessii/Collection1_cos_4kw.csv"
    model_save = "/disk/yjt/PersonalTarGuess/model/targuessii/t_collection1_4kw_m_targuessii.json"

    # popular_pwds = "/disk/yjt/PersonalTarGuess/data/popular/rockyou_demo.txt"
    # pwd_pairs = "/disk/data/targuess/2_train/targuessii/Collection1_cos_100.csv"
    # model_save = "/disk/yjt/PersonalTarGuess/model/targuessii/t_collection1_100_m_targuessii.json"

    model = TarGuessII()
    model.train_popular(popular_path=popular_pwds)
    model.train(pair_path=pwd_pairs)
    model.save(model_save)
    pass

if __name__ == '__main__':
    main()