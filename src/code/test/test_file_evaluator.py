import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from password_dataset import *
from seq2seq_model import *
from tokenizer import *
from pass2path_evaluator import *
import numpy as np
import itertools
import cProfile

PREDICT_DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# PREDICT_DEVICE = torch.device('cpu')

def main():
    model_load = "/disk/yjt/PersonalTarGuess/model/pass2path/t_collection_4kw_e_cos_m_pass2path_v4.pt"
    model = Pass2PathBeamSearchEvaluator(model_load, device=PREDICT_DEVICE)
    inputs = "/disk/data/targuess/3_query/Collection1_100k.csv"
    outputs = "/disk/yjt/PersonalTarGuess/result/csv/t_collection_4kw_q_Collection1_100k_m_pass2path_v4.csv"
    # inputs = "/disk/data/targuess/3_query/4iQ_100k.csv"
    # outputs = "/disk/yjt/PersonalTarGuess/result/csv/t_collection_4kw_q_4iQ_100k_m_pass2path_v3.csv"
    evaluator = FileEvaluator(model, outputs)
    evaluator.evaluate(inputs, 30, 1000, 64)
    evaluator.finish()

    print(f">>> Result saved in: {outputs}")


if __name__ == '__main__':
    # cProfile.run('main()', "/disk/yjt/PersonalTarGuess/profile_2.txt")
    main()