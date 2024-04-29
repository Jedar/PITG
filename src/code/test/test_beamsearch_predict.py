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

def main():
    model_load = "/disk/yjt/PersonalTarGuess/model/pass2path/t_collection_100_m_pass2path.pt"
    model = Pass2PathBeamSearchEvaluator(model_load, device=PREDICT_DEVICE)

    print(model.model)

    pwds = [
        "hello1", 
        "funtik44",
        "jebstone",
        "lerev1231", 
        "a12345"
    ]

    ans = model.predict(pwds)

    for items in ans:
        print(items)

if __name__ == '__main__':
    # main()
    cProfile.run('main()')