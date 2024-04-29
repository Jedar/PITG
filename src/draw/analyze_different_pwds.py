


dataset = "Collection1"
# dataset = "4iQ"

CONFIG = {
    "models":[
        {
            "label": "TarGuess-II",
            "path": f"/disk/yjt/PersonalTarGuess/result/csv/targuessii/t_collection1_4kw_q_{dataset}_100k_m_targuessii.csv",
            "color": "green",
            "line_style": "solid",
            "marker": ",",
            "marker_size": 2,
            "line_width": 1,
            "pwd_idx": 0,
            "guess_idx": 3
        },
        {
            "label": "PITG",
            "path": f"/disk/yjt/PersonalTarGuess/result/csv/pitg/t_collection_4kw_q_{dataset}_100k_m_pitg_v2.csv",
            "color": "blue",
            "line_style": "solid",
            "marker": ",",
            "marker_size": 2,
            "line_width": 1,
            "pwd_idx": 0,
            "guess_idx": 3
        },
        {
            "label": "PassBERT",
            "path": f"/disk/yjt/PasswordSimilarity/result/targuess/passbert/t_collection1_q_{dataset}_100k_m_passbert_4kw.csv",
            "color": "orange",
            "line_style": "solid",
            "marker": ",",
            "marker_size": 2,
            "line_width": 1,
            "pwd_idx": 0,
            "guess_idx": 3
        },
        {
            "label": "Pass2Path",
            "path": f"/disk/yjt/PasswordSimilarity/result/targuess/pass2path/t_collection1_q_{dataset}_100k_m_pass2path.csv",
            "color": "red",
            "line_style": "solid",
            "marker": ",",
            "marker_size": 2,
            "line_width": 1,
            "pwd_idx": 0,
            "guess_idx": 3
        },
        {
            "label": "Das",
            "path": f"/disk/yjt/PersonalTarGuess/result/csv/das/{dataset}_100k_das.csv",
            "color": "purple",
            "line_style": "solid",
            "marker": ",",
            "marker_size": 2,
            "line_width": 1,
            "pwd_idx": 0,
            "guess_idx": 3
        }
    ]
}

def read_model(model_name):
    config = None
    for item in CONFIG["models"]:
        if item["label"] == model_name:
            config = item
            break
    path = config["path"]
    guesses = {}
    with open(path, "r") as f:
        for line in f:
            ss = line.strip("\r\n").split("\t")
            gn = int(ss[config["guess_idx"]])
            if gn >= 0:
                guesses[(ss[0], ss[1])] = gn
    return guesses


def main():
    model1 = "PITG"
    model2 = "Das"

    guesses1 = read_model(model1)
    guesses2 = read_model(model2)

    K = set([k for k in guesses1.keys()] + [k for k in guesses2.keys()])

    G1 = []
    G2 = []

    for k in K:
        if k in guesses1 and k not in guesses2:
            G1.append((k, guesses1[k]))
        if k in guesses2 and k not in guesses1:
            G2.append((k, guesses2[k]))
    
    print(f">>> Passwords Cracked by Model1: ")
    for item in G1:
        print(item)
    print(f">>> Passwords Cracked by Model2: ")
    for item in G2:
        print(item)
    pass

if __name__ == '__main__':
    main()