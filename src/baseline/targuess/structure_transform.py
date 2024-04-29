from pcfg_basic import *
from edit_path import *

def pwd_pcfg_movement(x):
    items = pcfg_decompose(x)
    segments = [x[1] for x in items]
    segments = segments[1:] + [segments[0]]
    return "".join(segments)

def pwd_reverse(x):
    return x[::-1]

def pwd_segment_reverse(x):
    items = pcfg_decompose(x)
    segments = [x[1][::-1] for x in items]
    return "".join(segments)

TOTAL_TRANSFORMERS = ["C", "L", "SM", "R"]

TRANSFORMERS = {
    "C": ["C1", "C2", "C3", "C4"], 
    "L": ["L1", "L2", "L3", "L4", "L5"],
    "SM": ["SM"], 
    "R": ["R1", "R2"]
}

TRANSFORMER_FN = {
    "No": lambda x:x, 
    "C1" : lambda x:x.upper(), 
    "C2" : lambda x:x[0].upper()+x[1:], 
    "C3" : lambda x:x.lower(), 
    "C4" : lambda x:x[0].lower()+x[1:], 
    "L1": lambda x: x.replace("a", "@"), 
    "L2": lambda x: x.replace("s", "$"), 
    "L3": lambda x: x.replace("o", "0"),
    "L4": lambda x: x.replace("i", "1"),
    "L5": lambda x: x.replace("e", "3"), 
    "SM": pwd_pcfg_movement, 
    "R1": pwd_reverse, 
    "R2": pwd_segment_reverse
}

def get_closest_pwd(pwd1, pwd2, threshold):
    d = cosine_similarity(pwd1, pwd2)
    closest = pwd1
    livings = {
        "C": "No", 
        "L": "No", 
        "SM": "No",
        "R": "No"
    }
    for k in TRANSFORMERS.keys():
        step_best = closest
        step_d = d
        for fn in TRANSFORMERS[k]:
            # print(">>> transformers: ",k)
            pwd = TRANSFORMER_FN[fn](closest)
            pwd_d = cosine_similarity(closest, pwd2)
            if pwd_d > step_d:
                livings[k] = fn
                step_d = pwd_d
                step_best = pwd
        d = step_d
        closest = step_best
    if d < threshold:
        return False, closest, {"C": "No","L": "No","SM": "No","R": "No"}
    return True, closest, livings

