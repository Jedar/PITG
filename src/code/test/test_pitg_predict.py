import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from password_dataset import *
from pitg_model import *
from tokenizer import *
from pitg_evaluator import *
import numpy as np
import itertools
import cProfile

PREDICT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    model_load = "/disk/yjt/PersonalTarGuess/model/pitg/t_collection_4kw_e_cos_m_pitg_v2.pt"
    model = PITGBeamSearchEvaluator(model_load, device=PREDICT_DEVICE)
    inputs = "/disk/data/targuess/3_query/4iQ_10.csv"
    outputs = "/disk/yjt/PersonalTarGuess/result/csv/pitg/t_collection_4kw_q_4iQ_10_m_pitg_v2.csv"
    evaluator = FileEvaluator(model, outputs)
    evaluator.evaluate(inputs, 200, 1000, 64)
    evaluator.finish()

    print(f">>> Result saved in: {outputs}")

# def main():
#     model_load = "/disk/yjt/PersonalTarGuess/model/pitg/t_collection_100k_e_cos_m_pitg.pt"
#     model = PITGBeamSearchEvaluator(model_load, device=PREDICT_DEVICE)

#     print(model.model)

#     pwds = [
#         ("hello1", "lilcoach12345@yahoo.com"), 
#         ("funtik44", "elena44.114@yandex.ru"),
#         ("jebstone", "del7734@yahoo.com"),
#         ("lerev1231", "verelius@gmail.com"), 
#         ("a12345", "travel.m@hotmail.com"),
#         ("hello1", ""), 
#         ("funtik44", ""),
#         ("jebstone", ""),
#         ("lerev1231", ""), 
#         ("a12345", "")
#     ]

#     test = [x[0] for x in pwds]
#     emails = [parse_email(x[1]) for x in pwds]
#     regions = [x[2] for x in emails]
#     hosts = [x[1] for x in emails]

#     ans = model.predict(test, regions, hosts, beamwidth=20, topk=10)

#     for items in ans:
#         print(items)

if __name__ == '__main__':
    main()
    # cProfile.run('main()')