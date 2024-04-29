import matplotlib.pyplot as plt
import numpy as np 
import math
from matplotlib.lines import Line2D

plt.rcParams['font.family'] = 'SimHei'

dataset = "Collection1"
# dataset = "4iQ"

total_result = f"/disk/yjt/PersonalTarGuess/result/csv/pitg/t_collection_4kw_q_{dataset}_100k_m_region_v2_combine.csv"

demo_config = {
    "line_params":[
        {
            "label": "$PITG$",
            "path": total_result,
            "color": "black",
            "line_style": "--",
            "marker": "s",
            "marker_size": 2,
            "line_width": 1,
            "pwd_idx": 0,
            "guess_idx": 7
        },
        {
            "label": "$PITG_{ri+ui}$",
            "path": total_result,
            "color": "black",
            "line_style": "-.",
            "marker": "*",
            "marker_size": 2,
            "line_width": 1,
            "pwd_idx": 0,
            "guess_idx": 6
        },
        {
            "label": "$PITG_{ri+se}$",
            "path": total_result,
            "color": "black",
            "line_style": "-.",
            "marker": ".",
            "marker_size": 2,
            "line_width": 1,
            "pwd_idx": 0,
            "guess_idx": 5
        },
        {
            "label": "$PITG_{ri}$",
            "path": total_result,
            "color": "black",
            "line_style": ":",
            "marker": "v",
            "marker_size": 2,
            "line_width": 1,
            "pwd_idx": 0,
            "guess_idx": 4
        },
        {
            "label": "$PITG_{ce}$",
            "path": f"/disk/yjt/PersonalTarGuess/result/csv/pass2path/t_collection_4kw_q_{dataset}_100k_m_pass2path_v4.csv",
            "color": "black",
            "line_style": ":",
            "marker": "x",
            "marker_size": 2,
            "line_width": 1,
            "pwd_idx": 0,
            "guess_idx": 3
        },
        {
            "label": "$PITG_{hi}$",
            "path": f"/disk/yjt/PersonalTarGuess/result/csv/pitg/t_collection_4kw_q_{dataset}_100k_m_pitg_host_v3.csv",
            "color": "black",
            "line_style": ":",
            "marker": "o",
            "marker_size": 2,
            "line_width": 1,
            "pwd_idx": 0,
            "guess_idx": 3
        },
    ],
    "xlabel": "猜测数",
    "ylabel": "命中率 (%)",
    "label_weight": "normal",
    "label_size": 24,
    "xscale": "linear",
    "yscale": "linear",
    "legend_loc": "best",
    "legend_fontsize": 22,
    "tick_size": 22,
    "ylim_high": 100,
    "ylim_low": 0,
    "xlim_high": 20, 
    "xlim_low": 0,
    "xticks_val": [
            '$10^0$',
            '$10^1$',
            '$10^2$',
            '$10^3$'
            
    ],
    "yticks_val": [
            0,
            5,
            10,
            15,
            20,
            25
    ],
    "fig_size": "8 4",
    "fig_save": f"/disk/yjt/PersonalTarGuess/result/figure/{dataset}_inner_cracked.pdf"
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

def log_scale(X):
    return [math.log10(x) for x in X]

def draw(guess_stone,configs):
    mark_points = [1, 10, 100, 1000]
    marker_idx = [guess_stone.index(x) for x in mark_points]
    legend_args = []
    for line_para in configs["line_params"]:
        y = [c/line_para["total_count"] for c in line_para["count"]]
        plt.plot(log_scale(guess_stone),y,color = line_para["color"],label = line_para["label"],linestyle = line_para["line_style"])
        plt.plot(log_scale(mark_points),[y[i] for i in marker_idx],marker = line_para["marker"],color = line_para["color"],linestyle = 'none')
        legend_args.append(Line2D([], [], linestyle=line_para["line_style"], marker=line_para["marker"], color=line_para["color"], label=line_para["label"]))
    #纵坐标设置
    y = []
    for num in configs["yticks_val"]:
        temp = num/100
        y.append(temp)
    plt.yticks(y, configs["yticks_val"], fontsize=15)
    #横坐标设置
    x = configs["xticks_val"]
    tick_positions = np.linspace(0, 3, 4)
    #plt.xticks(x, [str(val) for val in x])
    plt.xticks(tick_positions,x, fontsize=15)

    # plt.xscale(configs["xscale"])
    # plt.yscale(configs["yscale"])

    plt.xlabel(configs["xlabel"], fontsize=16)
    plt.ylabel(configs["ylabel"], fontsize=16)
    
    plt.legend(loc="best", handles=legend_args, fontsize=14)
    plt.tight_layout()
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

