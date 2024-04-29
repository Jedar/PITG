from password_dataset import *
from pitg_model import *
from tokenizer import *
import numpy as np
import itertools
import tqdm

DEFAULT_DEVICE = torch.device('cpu')

class PITGEvaluator:
    def __init__(self, 
                 model_path, 
                 t1 = KBDPasswordTokenizer(), 
                 t2 = TransTokenizer(), 
                 t3 = RegionTokenizer(), 
                 t4 = HostTokenizer(), 
                 batch_size=32, 
                 max_len=16, 
                 device=DEFAULT_DEVICE):
        self.t1 = t1 
        self.t2 = t2
        self.t3 = t3 
        self.t4 = t4
        self.max_len = max_len
        self.model = torch.load(model_path).to(device)
        self.model.device = device
        self.device = device
        self.batch_size = batch_size
        self.model.set_mode("predict")
        self.model.eval()
    
    def predict(self, pwds):
        raise NotImplementedError()

class PITGGreedyEvaluator(PITGEvaluator):
    def __init__(self, 
                 model_path, 
                 t1=KBDPasswordTokenizer(), 
                 t2=TransTokenizer(), 
                 t3 = RegionTokenizer(), 
                 t4 = HostTokenizer(), 
                 batch_size=32, 
                 max_len=16, 
                 device=DEFAULT_DEVICE):
        super().__init__(model_path, t1, t2, t3, t4, batch_size, max_len, device)
        self.model.set_mode("predict")

    def _predict(self, pwds, regions, hosts):
        n = len(pwds)
        pwd_ids = self.t1(pwds, padding=False)
        pwd_ids = self.t1.padding(pwd_ids).to(self.device)
        regions = self.t3(regions).to(self.device)
        hosts = self.t4(hosts).to(self.device)
        digits = self.model(pwd_ids, regions, hosts)

        probs = [sum(x[1] for x in d) for d in digits]
        edits = [[x[0] for x in d] for d in digits]
        return zip(pwds,[self.t2.decode(pwds[i], edits[i]) for i in range(n)], probs)
    
    def predict(self, pwds, regions, hosts):
        n = len(pwds)
        combined = iter([])
        for i in range(0, n, self.batch_size):
            batch_x = pwds[i:i + self.batch_size]
            batch_y = regions[i:i + self.batch_size]
            batch_z = hosts[i:i + self.batch_size]
            combined = itertools.chain.from_iterable([combined, self._predict(batch_x, batch_y, batch_z)])
        return combined

class PITGBeamSearchEvaluator(PITGEvaluator):
    def __init__(self, 
                 model_path, 
                 t1=KBDPasswordTokenizer(), 
                 t2=TransTokenizer(), 
                 t3 = RegionTokenizer(), 
                 t4 = HostTokenizer(), 
                 batch_size=32, 
                 max_len=16, 
                 device=DEFAULT_DEVICE):
        super().__init__(model_path, t1, t2, t3, t4, batch_size, max_len, device)
        self.model.set_mode("beamsearch")

    def _predict(self, pwds, regions, hosts, beamwidth=10, topk=2):
        n = len(pwds)
        pwd_ids = self.t1(pwds, padding=False)
        pwd_ids = self.t1.padding(pwd_ids).to(self.device)
        regions = self.t3(regions).to(self.device)
        hosts = self.t4(hosts).to(self.device)
        
        digits = self.model(pwd_ids, regions, hosts, beamwidth=beamwidth, topk=topk)
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
    
    def predict(self, pwds, regions, hosts, beamwidth=10, topk=2):
        n = len(pwds)
        combined = iter([])
        for i in range(0, n, self.batch_size):
            batch_x = pwds[i:i + self.batch_size]
            batch_y = regions[i:i + self.batch_size]
            batch_z = hosts[i:i + self.batch_size]
            combined = itertools.chain.from_iterable([combined, self._predict(batch_x, batch_y, batch_z, beamwidth=beamwidth, topk=topk)])
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
    
    def parse_batch(self, inputs):
        src = [x[0] for x in inputs]
        tar = [x[1] for x in inputs]
        emails = [parse_email(x[2]) for x in inputs]
        regions = [x[2] for x in emails]
        hosts = [x[1] for x in emails]
        return src, tar, regions, hosts

    def evaluate(self, path, beamwidth=20, topk=1000, batch_size=64):
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
            src, tar, regions, hosts = self.parse_batch(inputs)
            outputs = list(self.model.predict(src, regions, hosts, beamwidth, topk))
            for item in zip(inputs, outputs):
                hit = False
                (src, tar, email), (_, output) = item
                cnt = 0
                for pwd, prob in output:
                    if pwd == tar:
                        hit = True
                        hit_count += 1
                        self.csv_out.write(f"{src}\t{tar}\t{email}\t{cnt}\t{prob}\n")
                        break
                    cnt += 1
                if not hit:
                    self.csv_out.write(f"{src}\t{tar}\t{email}\t{-1}\t{0.0}\n")
        print(f">>> Guess Rate: {hit_count / len(pwds)}")

    def finish(self):
        self.csv_out.close()

class HostFileEvaluator(FileEvaluator):
    def parse_batch(self, inputs):
        src = [x[0] for x in inputs]
        tar = [x[1] for x in inputs]
        emails = [parse_email(x[2]) for x in inputs]
        regions = [None for x in emails]
        hosts = [x[1] for x in emails]
        return src, tar, regions, hosts

class RegionFileEvaluator(FileEvaluator):
    def parse_batch(self, inputs):
        src = [x[0] for x in inputs]
        tar = [x[1] for x in inputs]
        emails = [parse_email(x[2]) for x in inputs]
        regions = [x[2] for x in emails]
        hosts = [None for x in emails]
        return src, tar, regions, hosts

def main():
    model_load = "/disk/yjt/PersonalTarGuess/model/pitg/t_collection_4kw_e_cos_m_pitg_v2.pt"
    model = PITGBeamSearchEvaluator(model_load, device=DEFAULT_DEVICE)
    # model = PITGGreedyEvaluator(model_load, device=DEFAULT_DEVICE)

    print(model.model)

    pwds = [
        ("hello1", "lilcoach12345@yahoo.com"), 
        ("funtik44", "elena44.114@yandex.ru"),
        ("jebstone", "del7734@yahoo.com"),
        ("lerev1231", "verelius@gmail.com"), 
        ("a12345", "travel.m@hotmail.com"),
        ("hello1", ""), 
        ("funtik44", ""),
        ("jebstone", ""),
        ("lerev1231", ""), 
        ("a12345", "")
    ]

    # pwds = [
    #     ("hello1", "lilcoach12345@yahoo.com"), 
    #     ("funtik44", "elena44.114@yandex.ru"),
    # ]

    test = [x[0] for x in pwds]
    emails = [parse_email(x[1]) for x in pwds]
    regions = [x[2] for x in emails]
    hosts = [x[1] for x in emails]

    ans = model.predict(test, regions, hosts, beamwidth=2, topk=5)
    # ans = model.predict(test, regions, hosts)

    for items in ans:
        pwd = items[0]
        predicts = items[1]
        print(f">>> {pwd} -> {len(predicts)}")
        print(items)
    
    # for items in ans:
    #     pwd = items[0]
    #     predicts = items[1]
    #     print(f">>> {pwd}")
    #     print(items)
        
if __name__ == '__main__':
    main()