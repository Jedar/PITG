from password_dataset import *
from pitg_model import *
from tokenizer import *
from pitg_preprocess import *
from pitg_evaluator import PITGBeamSearchEvaluator
import numpy as np
import itertools
import tqdm
import math
import json

DEFAULT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def log(x):
    if x <= 0 or x > 1:
        return 50
    return -math.log2(x)

def multiply(X, weight):
    Y = []
    for pwd, p in X:
        Y.append((pwd, log(weight) + p))
    return Y

class FileEvaluator:
    def __init__(self, model, weights_path, outputs):
        self.model = model
        self.csv_out = open(outputs, "w")
        with open(weights_path, "r") as f:
            args = json.load(f)
            self.weights = args["weights"]

    def generate_gn(self, pwd_list, tar_pwd):
        pwd_list = sorted(pwd_list, key=lambda x:x[1])
        for i, item in enumerate(pwd_list):
            pwd, prob = item
            if pwd == tar_pwd:
                return i, prob
        return -1, 0.0
    
    def structure_edits(self, src_pwd, ri):
        ans = []
        for fn in se_list:
            pwd = TRANSFORMER_FN[fn](src_pwd)
            ans.append((pwd, log(self.weights[ri][fn])))
        return ans
    
    def char_edits(self, src_pwd, host, ri, beamwidth, topk):
        outputs = list(self.model.predict([src_pwd], [ri], [host], beamwidth, topk))
        # consider credential stuffing attack
        return outputs[0][1] + [(src_pwd, 0)]
    
    def crack_pwds(self, pwd1, pwd2, email):
        ui, hi, ri = parse_email(email)
        # hi = None
        C1 = self.char_edits(pwd1, hi, ri, 200, 1000)
        C2 = self.structure_edits(pwd1, ri)
        C3 = self.char_edits(ui, hi, ri, 20, 50)
        C1 = multiply(C1, self.weights[ri]["alpha"])
        C3 = multiply(C3, self.weights[ri]["gamma"])

        print(C1)

        g1, _ = self.generate_gn(C1, pwd2)
        g2, _ = self.generate_gn(C1 + C2, pwd2)
        g3, _ = self.generate_gn(C1 + C3, pwd2)
        g4, p = self.generate_gn(C1 + C2 + C3, pwd2)
        return pwd1, pwd2, email, p, g1, g2, g3, g4
    
    def parse_batch(self, inputs):
        src = [x[0] for x in inputs]
        tar = [x[1] for x in inputs]
        emails = [parse_email(x[2]) for x in inputs]
        regions = [x[2] for x in emails]
        hosts = [x[1] for x in emails]
        usernames = [x[0] for x in emails]
        return src, tar, regions, hosts, usernames

    def evaluate(self, path, beamwidth=200, topk=1000, batch_size=64):
        hit_count = 0
        pwds = []
        batch_size = batch_size
        with open(path, "r") as f:
            for line in f:
                line = line.strip("\r\n").split("\t")
                src = line[0]
                target = line[1]
                email = line[2]
                pwds.append((src, target, email))
        for i in tqdm.tqdm(range(0, len(pwds), batch_size)):
            inputs = pwds[i:i+batch_size]
            src, tar, regions, hosts, usernames = self.parse_batch(inputs)
            outputs = list(self.model.predict(src, regions, hosts, beamwidth, topk))
            outputs2 = list(self.model.predict(usernames, regions, hosts, 20, 30))
            for item in zip(inputs, outputs, outputs2):
                hit = False
                (src, tar, email), (_, output), (_, ui_output) = item
                ui, hi, ri = parse_email(email)
                if ri not in self.model.t3.regions:
                    ri = ".com"
                C1 = output + [(src, 0)]
                C1 = multiply(C1, self.weights[ri]["alpha"])
                C2 = self.structure_edits(src, ri)
                C3 = ui_output + [(ui, 0)]
                C3 = multiply(C3, self.weights[ri]["gamma"])

                g1, _ = self.generate_gn(C1,tar)
                g2, _ = self.generate_gn(C1 + C2, tar)
                g3, _ = self.generate_gn(C1 + C3, tar)
                g4, p = self.generate_gn(C1 + C2 + C3, tar)
                hit_count += (1 if g4 > -1 else 0)
                self.csv_out.write(f"{src}\t{tar}\t{email}\t{p}\t{g1}\t{g2}\t{g3}\t{g4}\n")
        print(f">>> Guess Rate: {hit_count / len(pwds)}")
    
    # def evaluate(self, path):
    #     hit_count = 0
    #     pwds = []
    #     with open(path, "r") as f:
    #         for line in f:
    #             line = line.strip("\r\n").split("\t")
    #             src = line[0]
    #             target = line[1]
    #             email = line[2]
    #             pwds.append((src, target, email))
    #     for pwd1, pwd2, email in tqdm.tqdm(pwds):
    #         res = self.crack_pwds(pwd1, pwd2, email)
    #         hit_count += 1 if res[-1] > -1 else 0
    #         res = [str(x) for x in res]
    #         self.csv_out.write("\t".join(res)+"\n")
    #     print(f">>> Guess Rate: {hit_count / len(pwds)}")

    def finish(self):
        self.csv_out.close()

def main():
    model_load = "/disk/yjt/PersonalTarGuess/model/pitg/t_collection_4kw_e_cos_m_pitg_v2.pt"
    weights_load = "/disk/yjt/PersonalTarGuess/model/pitg_weights/Collection1_cos_4kw.json"
    model = PITGBeamSearchEvaluator(model_load, device=DEFAULT_DEVICE)
    inputs = "/disk/data/targuess/3_query/Collection_ru_10k.csv"
    result_save = "/disk/yjt/PersonalTarGuess/result/csv/pitg/t_collection_4kw_q_Collection_ru_10k_m_region_v2_combine.csv"

    evaluator = FileEvaluator(model, weights_load, result_save)
    evaluator.evaluate(inputs)
    evaluator.finish()

    print(f">>> Result saved in: {result_save}")
        
if __name__ == '__main__':
    main()