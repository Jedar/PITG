from collections import defaultdict
import json
import math

from pcfg_basic import *
from util import *
from edit_path import *
from structure_transform import *


class TarGuessIIEvaluator:
    def __init__(self, model_path, topk=1000):
        self.segment_map = None
        self.transformer_map = None
        self.structure_edit_map = None
        self.segment_edit_map = None
        self.popular_ratio = 0.0 
        self.popular_pwds = []
        self.topk = topk
        self.load(model_path, topk=self.topk)
        pass

    def _step1(self, pwd):
        for k in self.transformer_map.keys():
            for x, y in self.transformer_map[k]:
                pwd1 = TRANSFORMER_FN[x](pwd)
                yield (pwd1, log(y))

    def _step2(self, pwd, max_step=10, max_generated=20):
        pcfg_items = pcfg_decompose(pwd)
        segments = [tag_to_str(x[1]) for x in pcfg_items]
        template = [tag_to_str(x[0]) for x in pcfg_items]
        template_str = "".join(template)
        if template_str not in self.structure_edit_map:
            return []
        M = self.structure_edit_map[template_str]
        if "No" in M:
            yield (pwd,  M["No"][0])
        if "hd" in M:
            yield combine_segment(segments[1:]), M["hd"][0]
        if "td" in M:
            yield combine_segment(segments[:-1]), M["td"][0]
        total = 0
        if "hi" in M:
            p1 = log(M["hi"][0])
            for item in M["hi"][1]:
                tag = item[0]
                p2 = log(item[1])
                if tag not in self.segment_map:
                    continue
                cnt = 0
                for ss, p3 in self.segment_map[tag]:
                    p3 = log(p3)
                    yield combine_segment([ss] + segments), p1+p2+p3
                    cnt += 1
                    if cnt > max_step:
                        break
                total += cnt 
                if total > max_generated:
                    break
        if "ti" in M:
            p1 = log(M["ti"][0])
            for item in M["ti"][1]:
                tag = item[0]
                p2 = log(item[1])
                # print(self.segment_map)
                if tag not in self.segment_map:
                    continue
                cnt = 0
                for ss, p3 in self.segment_map[tag]:
                    p3 = log(p3)
                    yield combine_segment(segments + [ss]), p1+p2+p3
                    cnt += 1
                    if cnt > max_step:
                        break
                total += cnt 
                if total > max_generated:
                    break
    
    def _generate_word(self, template, word, max_step=3):
        if template in self.segment_edit_map:
            M = self.segment_edit_map[template]
            if "No" in M:
                yield word, log(M["No"][0])
            if "hd" in M:
                yield word[1:], log(M["hd"][0])
            if "td" in M:
                yield word[:-1], log(M["td"][0])
            cnt = 0
            if "ti" in M:
                p1 = log(M["ti"][0])
                for ch, p2 in M["ti"][1]:
                    yield word+ch, p1+log(p2)
                    cnt += 1
                    if cnt > max_step:
                        break
            cnt = 0
            if "hi" in M:
                p1 = log(M["hi"][0])
                for ch, p2 in M["hi"][1]:
                    yield ch + word, p1+log(p2)
                    cnt += 1
                    if cnt > max_step:
                        break

    def _generate_combine_string(self, arr):
        if len(arr) == 0:
            return []
        if len(arr) == 1:
            for s, p in arr[0]:
                yield s, p
        else:
            current = arr[0]
            remain = arr[1:]
            for s, p1 in current:
                for combine, p2 in self._generate_combine_string(remain):
                    yield s + combine, p1+p2

    def _step3(self, pwd, max_step=30, max_generated=30):
        pcfg_items = pcfg_decompose(pwd)
        segments = [tag_to_str(x[1]) for x in pcfg_items]
        template = [tag_to_str(x[0]) for x in pcfg_items]
        n = len(segments)
        S = [self._generate_word(template[i], segments[i], max_step=max_step) for i in range(n)]
        cnt = 0
        for pwd, p in self._generate_combine_string(S):
            yield pwd, p
            cnt += 1
            if cnt > max_generated:
                break

    def _generate_candidate(self, pwd):
        for w1, p1 in self._step1(pwd):
            for w2, p2 in self._step2(w1):
                for w3, p3 in self._step3(w2):
                    yield w3, p1+p2+p3

    def generate_guess(self, pwd, max_generated=1000):
        pwds = []
        candidates = self._generate_candidate(pwd)
        for i in range(max_generated):
            values = next(candidates, None)
            if values:
                pwds.append(values)
            else:
                break
        pwds = sorted(pwds+self.popular_pwds, key=lambda x:x[1])
        if len(pwds) > max_generated:
            pwds = pwds[:max_generated]
        return pwds
    
    def evaluate_file(self, input_file, output_file, max_generated=1000):
        pair_cnt = count_lines(input_file)
        print(f">>> Load password pairs: {input_file}, Size: {pair_cnt}")

        pairs = read_pwd_pairs(input_file, with_bar=True)
        hit_cnt = 0

        with open(output_file, "w") as f:
            for pair in pairs:
                source, target = pair 
                guesses = self.generate_guess(source, max_generated=max_generated)
                hit = False
                for i, (pwd, prob) in enumerate(guesses):
                    if pwd == target:
                        hit = True 
                        f.write(f"{source}\t{target}\t{prob}\t{i}\n")
                        hit_cnt += 1
                        break
                if not hit:
                    f.write(f"{source}\t{target}\t{0.0}\t{-1}\n")
        
        print(f">>> Accuracy: {hit_cnt / pair_cnt}")
        print(f">>> Result saved in: {output_file}")

    def load(self, model_path, topk=1000):
        print(f">>> Load TarGuess II Model from {model_path}")
        with open(model_path, "r") as f:
            obj = json.load(f)
            self.segment_map = obj["segment_map"]
            self.transformer_map = obj["transformer_map"]
            self.structure_edit_map = obj["structure_edit_map"]
            self.segment_edit_map = obj["segment_edit_map"]
            self.popular_ratio = log(obj["popular_ratio"])
            self.popular_pwds = obj["popular_pwds"]

            # Reduce popular password size
            if len(self.popular_pwds) > topk:
                self.popular_pwds = self.popular_pwds[:topk]
                self.popular_pwds = [(x, y+self.popular_ratio) for x, y in self.popular_pwds]


def main():
    model_load = "/disk/yjt/PersonalTarGuess/model/targuessii/t_collection1_4kw_m_targuessii.json"
    model = TarGuessIIEvaluator(model_load)

    # input = "/disk/data/targuess/3_query/pair/4iQ_100k.csv"
    # output = "/disk/yjt/PersonalTarGuess/result/csv/targuessii/t_collection1_4kw_q_4iQ_100k_m_targuessii.csv"

    input = "/disk/data/targuess/3_query/pair/Collection1_fr_10k.csv"
    output = "/disk/yjt/PersonalTarGuess/result/csv/targuessii/t_collection1_4kw_q_Collection1_fr_10k_m_targuessii.csv"

    model.evaluate_file(input, output)

if __name__ == '__main__':
    main()