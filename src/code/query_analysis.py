from util import *

class Stats:
    def __init__(self):
        self.same = 0
        self.total = 0
        self.username_same = 0
        pass

    def parse(self, pwd1, pwd2, email):
        self.total += 1
        if pwd1 == pwd2:
            self.same += 1
        username, host, region = parse_email(email)
        if username == pwd2:
            self.username_same += 1
    
    def show(self):
        print(f">>> Total: {self.total}")
        print(f">>> Same ratio: {self.same / self.total}")
        print(f">>> Same Username: {self.username_same / self.total}")

def main():
    path = "/disk/data/targuess/3_query/Collection1_100k.csv"

    print(f">>> Read passwords: {path}")
    pwds = read_query(path)
    stats = Stats()
    for pwd1, pwd2, email in pwds:
        stats.parse(pwd1, pwd2, email)
    stats.show()
    pass

if __name__ == '__main__':
    main()