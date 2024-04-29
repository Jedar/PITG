import matplotlib.pyplot as plt
import numpy as np
from functools import lru_cache
import tqdm

plt.rcParams['font.family'] = 'SimHei'

dataset = "Collection1"
# dataset = "4iQ"

demo_config = {
    "line_params":[
        {
            "label": "PITG",
            "y": {
                "Collection1": [23.47,25.34,25.25,27.40,29.82], 
                "4iQ": [20.94, 25.05, 22.99, 30.03, 22.59]
            },
            "color": "grey",
            "edgecolor": "black", 
            "hatch": "x",
            "width": 0.15,
            "pwd_idx": 0,
            "guess_idx": 7
        },
        {
            "label": "PITG$_{ri}$",
            "y": {
                "Collection1": [18.66,20.80,20.74,18.94,24.95], 
                "4iQ": [18.05, 17.01, 20.97, 25.05, 19.32]
            },
            "color": "white",
            "edgecolor": "black", 
            "hatch": "O",
            "width": 0.15,
            "pwd_idx": 0,
            "guess_idx": 4
        },
        {
            "label": "PassBERT",
            "y": {
                "Collection1": [18.62,19.43,19.48,19.65,24.82], 
                "4iQ": [17.87, 16.03, 20.13, 25.25, 19.05]
            },
            "color": "lightgrey",
            "edgecolor": "black", 
            "hatch": "-",
            "width": 0.15,
            "pwd_idx": 0,
            "guess_idx": 3
        },
        {
            "label": "TarGuess",
            "y": {
                "Collection1": [11.18, 13.74, 14.07, 10.52, 17.90], 
                "4iQ": [13.03, 13.56, 13.78, 16.93, 16.66]
            },
            "color": "whitesmoke",
            "edgecolor": "black", 
            "hatch": "/",
            "width": 0.15,
            "pwd_idx": 0,
            "guess_idx": 3
        }
    ],
    "xlabel": "编辑距离",
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
    "items": 5, 
    "xticks_val": ["整体", ".cn", ".uk", ".ru", ".fr"],
    "yticks_val": [
            0,
            5,
            10,
            15,
            20,
            25
    ],
    "fig_size": (8, 4),
    "fig_save": f"/disk/yjt/PersonalTarGuess/result/figure/{dataset}_region_analysis.pdf"
}

def count_lines(file_path):
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

def save_grouped_bar_chart(line_params, save_path):
    # 设置图形大小
    plt.figure(figsize=line_params["fig_size"])

    data = line_params["line_params"]
    bar_width = 0.15
    gap_width = 0.25
    item_cnt = line_params["items"]
    num_groups = len(data)
    group_width = bar_width * num_groups
    index = np.arange(item_cnt)*(group_width + gap_width) - (num_groups / 2.0)*bar_width

    # 绘制柱状图
    for i, d in enumerate(data):
        values = d["y"][dataset]
        plt.bar(index+i*bar_width, values, color=d["color"], edgecolor=d["edgecolor"], hatch=d["hatch"], width=d["width"], align='center')

    # 设置x轴刻度和标签
    plt.xticks(index + (num_groups / 2.0 - 0.5)*bar_width, line_params["xticks_val"], fontsize=15)
    plt.yticks(fontsize=15)

    plt.xlabel(line_params["xlabel"], fontsize=16)
    plt.ylabel(line_params["ylabel"], fontsize=16)

    plt.ylim((0, 40))

    # 添加图例
    legend_labels = [params['label'] for params in data]
    plt.legend(legend_labels, loc="upper left", fontsize=15, ncol=4)
    plt.tight_layout()

    # 保存图形到指定路径
    plt.savefig(save_path)

    print(f">>> Figure saved in {save_path}")

    # 关闭图形
    plt.close()

def main():
    config = demo_config
    
    save_grouped_bar_chart(config, config["fig_save"])


main()