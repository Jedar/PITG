import matplotlib.pyplot as plt
import numpy as np 

dataset = "Collection1"
# dataset = "4iQ"

demo_config = {
    "line_params":[
        {
            "label": "TarGuess II",
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
    ],
    "xlabel": "Guesses",
    "ylabel": "Cracked (%)",
    "label_weight": "normal",
    "label_size": 24,
    "xscale": "log",
    "yscale": "linear",
    "legend_loc": "best",
    "legend_fontsize": 22,
    "tick_size": 22,
    "ylim_high": 100,
    "ylim_low": 0,
    "xlim_high": 20, 
    "xlim_low": 0,
    "xticks_val": [
            0,
            10,
            100,
            1000
            
    ],
    "yticks_val": [
            0,
            5,
            10,
            15,
            20,
            25
    ],
    "fig_size": "8 4.95",
    "fig_save": f"/disk/yjt/PersonalTarGuess/result/figure/{dataset}_cracked.pdf"
}


def rate_count(config):
    guess_stone = list(range(1,1001,1))

    res_dict = {}
    path = config["path"]
    count = [0 for x in range(len(guess_stone))]
    total_count = 0
    guess_idx = config["guess_idx"]

    with open(path, "r") as f:
        for line in f:
            total_count += 1
            line = line.strip("\r\n")
            ss = line.split("\t")
            value = int(ss[guess_idx])
            # total_count += 1
            if value == -1:
                continue
                # if value <= 10:
                #     print(ss[0])
            for i, stone in enumerate(guess_stone):
                if value < stone:
                    count[i] += 1
    print("Targuess Counter")
    print(f"Total count: {total_count}")

    config["count"] = count
    config["total_count"] = total_count
    return config


def draw(guess_stone,configs):
    for line_para in configs["line_params"]:

        y = [c/line_para["total_count"] for c in line_para["count"]]
        plt.plot(guess_stone,y,marker = ',',color = line_para["color"],label = line_para["label"],linestyle = line_para["line_style"])
    #纵坐标设置
    y = []
    for num in configs["yticks_val"]:
        temp = num/100
        y.append(temp)
    plt.yticks(y, configs["yticks_val"])
    #横坐标设置
    x = configs["xticks_val"]
    tick_positions = np.linspace(0, 3, 4)
    #plt.xticks(x, [str(val) for val in x])
    plt.xticks(tick_positions,x)

    plt.xscale(configs["xscale"])
    plt.yscale(configs["yscale"])

    plt.grid(True)
    plt.xlabel(configs["xlabel"])
    plt.xlabel(configs["ylabel"])
    plt.title("Guess")
    
    
    plt.legend()
    plt.savefig(configs["fig_save"])
    save_path = configs["fig_save"]

    print(f">>> Result saved in :{save_path}")


def main():
    for i,config in enumerate(demo_config["line_params"]):
        demo_config["line_params"][i] = rate_count(config)

    guess_stone = list(range(1,1001,1))

    draw(guess_stone,demo_config)
    pass



if __name__ == '__main__':
    main()

