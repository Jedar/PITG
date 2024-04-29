from password_dataset import *
from seq2seq_model import *
from tokenizer import *
import numpy as np
import itertools
import tqdm

DEFAULT_DEVICE = torch.device('cpu')

class Pass2PathEvaluator:
    def __init__(self, model_path, t1=KBDPasswordTokenizer(), t2=TransTokenizer(), batch_size=32, max_len=16, device=DEFAULT_DEVICE):
        self.t1 = t1 
        self.t2 = t2
        self.max_len = max_len
        self.model = torch.load(model_path).to(device)
        self.model.device = device
        self.device = device
        self.batch_size = batch_size
        self.model.set_mode("predict")
        self.model.eval()
    
    def predict(self, pwds):
        raise NotImplementedError()

class Pass2PathGreedyEvaluator(Pass2PathEvaluator):
    def __init__(self, model_path, t1=KBDPasswordTokenizer(), t2=TransTokenizer(), batch_size=32, max_len=16, device=DEFAULT_DEVICE):
        super().__init__(model_path, t1, t2, batch_size, max_len, device)
        self.model.set_mode("predict")

    def _predict(self, pwds):
        n = len(pwds)
        pwd_ids = self.t1(pwds, padding=False)
        pwd_ids = self.t1.padding(pwd_ids).to(self.device)
        digits = self.model(pwd_ids)
        
        probs = [sum(x[1] for x in d) for d in digits]
        edits = [[x[0] for x in d] for d in digits]
        return zip(pwds,[self.t2.decode(pwds[i], edits[i]) for i in range(n)], probs)
    
    def predict(self, pwds):
        n = len(pwds)
        combined = iter([])
        for i in range(0, n, self.batch_size):
            batch = pwds[i:i + self.batch_size]
            combined = itertools.chain.from_iterable([combined, self._predict(batch)])
        return combined

class Pass2PathBeamSearchEvaluator(Pass2PathEvaluator):
    def __init__(self, model_path, t1=KBDPasswordTokenizer(), t2=TransTokenizer(), batch_size=32, max_len=16, device=DEFAULT_DEVICE):
        super().__init__(model_path, t1, t2, batch_size, max_len, device)
        self.model.set_mode("beamsearch")

    def _predict(self, pwds, beamwidth=10, topk=2):
        n = len(pwds)
        pwd_ids = self.t1(pwds, padding=False)
        x_length = torch.tensor([len(x) for x in pwd_ids])
        pwd_ids = self.t1.padding(pwd_ids).to(self.device)
        
        digits = self.model(pwd_ids, x_length, beamwidth=beamwidth, topk=topk)
        ans = []
        for i in range(n):
            items = []
            outputs = digits[i]
            for output in outputs:
                prob = output[1]
                edits = output[0]
                pwd = self.t2.decode(pwds[i], edits)
                items.append((pwd, prob))
            ans.append(items)
        return zip(pwds,ans)
    
    def predict(self, pwds, beamwidth=10, topk=5):
        n = len(pwds)
        combined = iter([])
        for i in range(0, n, self.batch_size):
            batch = pwds[i:i + self.batch_size]
            combined = itertools.chain.from_iterable([combined, self._predict(batch, beamwidth=beamwidth, topk=topk)])
        return combined
    
    def predict_one(self, pwd, beamwidth=10, topk=2):
        return next(self._predict([pwd], beamwidth=beamwidth, topk=topk))

class FileEvaluator:
    def __init__(self, model,outputs):
        self.model = model
        self.csv_out = open(outputs, "w")
        
    def count_lines(self, file_path):
        line_count = 0
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    line_count += 1
        except FileNotFoundError:
            print("文件不存在：", file_path)
        except IOError:
            print("无法打开文件：", file_path)
        return line_count

    def evaluate(self, path, beamwidth=20, topk=1000, batch_size=64):
        hit_count = 0
        pwds = []
        batch_size = batch_size
        with open(path, "r") as f:
            for line in f:
                line = line.strip("\r\n").split("\t")
                src = line[0]
                target = line[1]
                pwds.append((src, target))
        for i in tqdm.tqdm(range(0, len(pwds), batch_size)):
            inputs = pwds[i:i+batch_size]
            src = [x[0] for x in inputs]
            outputs = list(self.model.predict(src, beamwidth, topk))
            for item in zip(inputs, outputs):
                hit = False
                (src, tar), (_, output) = item
                cnt = 0
                for pwd, prob in output:
                    if pwd == tar:
                        hit = True
                        hit_count += 1
                        self.csv_out.write(f"{src}\t{tar}\t{prob}\t{cnt}\n")
                        break
                    cnt += 1
                if not hit:
                    self.csv_out.write(f"{src}\t{tar}\t{0.0}\t{-1}\n")
        print(f">>> Guess Rate: {hit_count / len(pwds)}")

    def finish(self):
        self.csv_out.close()
    

def main():
    model_load = "/disk/yjt/PersonalTarGuess/model/pass2path/t_collection_4bw_e_cos_m_pass2path_v3.pt"
    model = Pass2PathGreedyEvaluator(model_load)

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
    main()