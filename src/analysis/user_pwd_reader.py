import json

topk = 10
topv = 4

json_path= "/disk/yjt/PersonalTarGuess/result/analysis/Collections1_1bw_user_pwd.json"
# /disk/yjt/PersonalTarGuess/src/analysis/user_pwd_reader.py
# json_path= "/disk/yjt/PersonalTarGuess/result/analysis/4iQ_user_pwd_behavior.json"

region_keys = [".com", ".ru", ".fr", ".uk", ".cn", ".jp"]
host_keys = ["@hotmail", "@yahoo", "@gmail", "@msn", "@qq", "@163"]

def count(list):
    return sum([int(v) for v in list.values()])

def main():
    with open(json_path, "r") as f:
        values = json.load(f)

    keys = [str(i) for i in range(topv)]
    
    print(">>> Total: ")
    v = values["total_cnt"]
    total = count(v)
    for k in keys:
        if k not in v:
            v[k] = 0
        print(f">>> {k}: {v[k]} - {int(v[k]) / total}")
    
    print(">>> Region: ")
    # items = [k for k in values["region_cnt"].keys()][:topk]
    items = region_keys
    for i in items:
        print(f">>> {i}")
        v = values["region_cnt"][i]
        total = count(v)
        for k in keys:
            if k not in v:
                v[k] = 0
            print(f">>> {k}: {v[k]} - {int(v[k]) / total}")


    # print(">>> Host: ")
    # # items = [k for k in values["host_cnt"].keys()][:topk]
    # items = [x[1:] for x in host_keys]
    # for i in items:
    #     print(f">>> {i}")
    #     if i not in values["host_cnt"]:
    #         continue 
    #     v = values["host_cnt"][i]

    #     total = count(v)
    #     for k in keys:
    #         if k not in v:
    #             v[k] = 0
    #         print(f">>> {k}: {v[k]} - {int(v[k]) / total}")
    pass

main()