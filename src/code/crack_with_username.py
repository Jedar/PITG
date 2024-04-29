from util import *

class Stats:
    def __init__(self):
        self.same = 0
        self.total = 0
        self.username_same = 0
        self.guess_stone = [5, 10, 100, 1000]
        self.count = [0, 0, 0, 0]
        pass

    def _real_gn(self, pwd1, pwd2, email, gn):
        if pwd1 == pwd2:
            return 0
        username, _, _ = parse_email(email)
        if username == pwd2:
            return 1
        if gn == -1:
            return -1
        return gn + 2
    
    def _with_stuffing_gn(self, pwd1, pwd2, email, gn):
        if pwd1 == pwd2:
            return 0
        # username, _, _ = parse_email(email)
        # if username == pwd2:
        #     return 1
        if gn == -1:
            return -1
        return gn + 1

    def parse(self, pwd1, pwd2, email, gn):
        self.total += 1
        gn = self._with_stuffing_gn(pwd1, pwd2, email, gn)
        if gn == -1:
            return 
        for i, stone in enumerate(self.guess_stone):
            if gn < stone:
                self.count[i] += 1
    
    def show(self):
        print(f">>> Total: {self.total}")
        for i, value in enumerate(self.count):
            print(f"Guess number {self.guess_stone[i]}: {value / self.total}({value})")

def read_cracking_result(file_path):
    pwds = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip("\r\n").split("\t")
            src = line[0]
            target = line[1]
            email = line[2]
            gn = int(line[3])
            prob = float(line[4])
            pwds.append((src, target, email, gn))
    return pwds

def main():
    path = "/disk/yjt/PersonalTarGuess/result/csv/pitg/t_collection_4kw_q_Collection1_100k_m_pitg_v2.csv"

    print(f">>> Read passwords: {path}")
    stats = Stats()
    pwds = read_cracking_result(path)
    for pwd1, pwd2, email, gn in pwds:
        stats.parse(pwd1, pwd2, email, gn)
    stats.show()
    pass

if __name__ == '__main__':
    main()