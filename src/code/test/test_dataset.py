from tokenizer import *
from password_dataset import *

data_path = "/disk/data/targuess/2_train/pitg/Collection1_cos_100.csv"

t1 = KBDPasswordTokenizer()
t2 = TransTokenizer()

print(f">>> Load data from {data_path}")

dataset = pass2path_dataloader(data_path, t1, t2, 2, False)

for i in dataset:
    print(i)
    pass