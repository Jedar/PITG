import json

topk = 20
topv = 6

# json_path= "/disk/yjt/PersonalTarGuess/result/analysis/4iQ_user_edit_behavior.json"

json_path= "/disk/yjt/PersonalTarGuess/result/analysis/Collections1_user_edit_behavior.json"

region_keys = [".com", ".ru", ".fr", ".uk", ".cn", ".jp"]
host_keys = ["@hotmail", "@yahoo", "@gmail", "@msn", "@qq", "@163"]

def main():
    with open(json_path, "r") as f:
        values = json.load(f)
    
    # print(">>> Region: ")
    # # keys = [k for k in values["region"].keys()][:topk]
    # keys = region_keys
    # for key in keys:
    #     print(f">>> {key}")
    #     value = values["region"][key]
    #     value = value[:topv]
    #     for v in value:
    #         print(f">>> {v}")


    # print(">>> Host: ")
    # keys = [x[1:] for x in host_keys]
    # for key in keys:
    #     print(f">>> {key}")
    #     value = values["host"][key]
    #     value = value[:topv]
    #     for v in value:
    #         print(f">>> {v}")

    keys = values["host"].keys()
    X = []
    for x in keys:
        if x in X:
            continue
        flag = False
        for y in X:
            if y in x:
                 flag = True
        if not flag:
            X.append(x)
    print(X)
    pass

main()