from torch.utils.data import Dataset, DataLoader
import json
import csv
import torch

from tokenizer import *
from util import *

class PITGPTrainDataset(Dataset):
    def __init__(self, path, max_len=30):
        self.data = []
        with open(path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                p1, p2, email, path = row[0], row[1], row[2], json.loads(row[3])
                if len(p1) > max_len:
                    continue
                self.data.append((p1, p2, email, path))
        
    def __getitem__(self, index):
        return {
            "pass1": self.data[index][0], 
            "pass2": self.data[index][1],
            "email": self.data[index][2],
            "edit": self.data[index][3]
        }
    
    def __len__(self):
        return len(self.data)


#修改处在这里
class Pass2PathDataset(Dataset):
    def __init__(self, path):
        self.data = []
        with open(path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 2:
                    continue
                p1, p2, path = row[0], row[1], json.loads(row[2])
                self.data.append((p1, p2, path))
    
    def __getitem__(self, index):
        return {
            "pass1": self.data[index][0], 
            "pass2": self.data[index][1],
            "edit": self.data[index][2]
        }
    
    def __len__(self):
        return len(self.data)

class Pass2PathCollator:
    def __init__(self, pwd_tokenizer, path_tokenizer, max_len=16):
        self.t1 = pwd_tokenizer
        self.t2 = path_tokenizer
        self.max_len = max_len

    def padding(self, seqs, pad_id):
        max_length = max([len(seq) for seq in seqs])
        arr =  [seq + [pad_id] * (max_length - len(seq)) for seq in seqs]
        return torch.tensor(arr)
    
    def __call__(self, batch):
        pwds = [x["pass1"] for x in batch]
        paths = [x["edit"] for x in batch]
        pwds = [self.t1.encode(x) for x in pwds]
        x_length = [len(x) for x in pwds]
        x_length = torch.tensor(x_length)
        # 尾部增加终止符
        paths = [x  + [self.t2.end_token_id] for x in paths]
        pwds = self.padding(pwds, self.t1.pad_token_id)
        paths = self.padding(paths, self.t2.pad_token_id)
        return pwds, paths

class PITGCollator:
    def __init__(self, pwd_tokenizer, path_tokenizer, region_tokenizer, host_tokenizer, max_len=16):
        self.t1 = pwd_tokenizer
        self.t2 = path_tokenizer
        self.t3 = region_tokenizer
        self.t4 = host_tokenizer
        self.max_len = max_len

    def padding(self, seqs, pad_id):
        max_length = max([len(seq) for seq in seqs])
        arr =  [seq + [pad_id] * (max_length - len(seq)) for seq in seqs]
        return torch.tensor(arr)
    
    def __call__(self, batch):
        pwds = [x["pass1"] for x in batch]
        paths = [x["edit"] for x in batch]
        pwds = [self.t1.encode(x) for x in pwds]
        emails = [parse_email(x["email"]) for x in batch]
        regions = torch.tensor([self.t3.encode(x[2]) for x in emails])
        hosts = torch.tensor([self.t4.encode(x[1]) for x in emails])
        # 尾部增加终止符
        paths = [x  + [self.t2.end_token_id] for x in paths]
        pwds = self.padding(pwds, self.t1.pad_token_id)
        paths = self.padding(paths, self.t2.pad_token_id)
        return pwds, regions, hosts, paths

class PITGHostCollator(PITGCollator):
    def __call__(self, batch):
        pwds = [x["pass1"] for x in batch]
        paths = [x["edit"] for x in batch]
        pwds = [self.t1.encode(x) for x in pwds]
        emails = [parse_email(x["email"]) for x in batch]
        regions = torch.tensor([self.t3.encode(None) for x in emails])
        hosts = torch.tensor([self.t4.encode(x[1]) for x in emails])
        # 尾部增加终止符
        paths = [x  + [self.t2.end_token_id] for x in paths]
        pwds = self.padding(pwds, self.t1.pad_token_id)
        paths = self.padding(paths, self.t2.pad_token_id)
        return pwds, regions, hosts, paths

class PITGRegionCollator(PITGCollator):
    def __call__(self, batch):
        pwds = [x["pass1"] for x in batch]
        paths = [x["edit"] for x in batch]
        pwds = [self.t1.encode(x) for x in pwds]
        emails = [parse_email(x["email"]) for x in batch]
        regions = torch.tensor([self.t3.encode(x[2]) for x in emails])
        hosts = torch.tensor([self.t4.encode(None) for x in emails])
        # 尾部增加终止符
        paths = [x  + [self.t2.end_token_id] for x in paths]
        pwds = self.padding(pwds, self.t1.pad_token_id)
        paths = self.padding(paths, self.t2.pad_token_id)
        return pwds, regions, hosts, paths

def pass2path_dataloader(path, t1, t2, batch_size=256, shuffle=True):
    dataset = PITGPTrainDataset(path)
    collator = Pass2PathCollator(t1, t2)
    return DataLoader(dataset, batch_size, shuffle, collate_fn=collator)

def pitg_dataloader(path, t1, t2, t3, t4, batch_size=256, shuffle=True):
    dataset = PITGPTrainDataset(path)
    collator = PITGCollator(t1, t2, t3, t4)
    return DataLoader(dataset, batch_size, shuffle, collate_fn=collator)

def pitg_region_dataloader(path, t1, t2, t3, t4, batch_size=256, shuffle=True):
    print(">>> PITG Region Dataloader")
    dataset = PITGPTrainDataset(path)
    collator = PITGRegionCollator(t1, t2, t3, t4)
    return DataLoader(dataset, batch_size, shuffle, collate_fn=collator)

def pitg_host_dataloader(path, t1, t2, t3, t4, batch_size=256, shuffle=True):
    print(">>> PITG Host Dataloader")
    dataset = PITGPTrainDataset(path)
    collator = PITGHostCollator(t1, t2, t3, t4)
    return DataLoader(dataset, batch_size, shuffle, collate_fn=collator)