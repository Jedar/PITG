import matplotlib.pyplot as plt
import numpy as np
from functools import lru_cache
import tqdm
from collections import defaultdict
import math

plt.rcParams['font.family'] = 'SimHei'

dataset = "Collection1"
# dataset = "4iQ"

total_result = f"/disk/yjt/PersonalTarGuess/result/csv/pitg/t_collection_4kw_q_{dataset}_100k_m_region_v2_combine.csv"

target = f"/disk/data/targuess/3_query/{dataset}_100k.csv"

n = 5

demo_config = {
    "line_params":[
        {
            "label": "PITG",
            "path": total_result,
            "color": "grey",
            "edgecolor": "black", 
            "hatch": "x",
            "width": 0.15,
            "pwd_idx": 0,
            "guess_idx": 7
        },
        {
            "label": "PITG$_{ri}$",
            "path": total_result,
            "color": "white",
            "edgecolor": "black", 
            "hatch": "O",
            "width": 0.15,
            "pwd_idx": 0,
            "guess_idx": 4
        },
        {
            "label": "PassBERT",
            "path": f"/disk/yjt/PasswordSimilarity/result/targuess/passbert/t_collection1_q_{dataset}_100k_m_passbert_4kw.csv",
            "color": "lightgrey",
            "edgecolor": "black", 
            "hatch": "-",
            "width": 0.15,
            "pwd_idx": 0,
            "guess_idx": 3
        },
        {
            "label": "TarGuess-II",
            "path": f"/disk/yjt/PersonalTarGuess/result/csv/targuessii/t_collection1_4kw_q_{dataset}_100k_m_targuessii.csv",
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
    "xticks_val": [str(i) for i in range(1, n+1)] + [f">{n}"],
    "yticks_val": [
            0,
            5,
            10,
            15,
            20,
            25
    ],
    "fig_size": (8, 4),
    "fig_save": f"/disk/yjt/PersonalTarGuess/result/figure/{dataset}_edit_distance_analysis.pdf"
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

@lru_cache(maxsize=200000)
def edit_distance(str1, str2):
    n = len(str1)
    m = len(str2)
    D = np.full((n + 1, m + 1), np.inf)
    trace = np.full((n + 1, m + 1), None)
    trace[1:, 0] = list(zip(range(n), np.zeros(n, dtype=int)))
    trace[0, 1:] = list(zip(np.zeros(m, dtype=int), range(m)))
    # Initialization:
    D[:, 0] = np.arange(n + 1)
    D[0, :] = np.arange(m + 1)
    # Fill the matrices:
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            delete = D[i - 1, j] + 1
            insert = D[i, j - 1] + 1
            if (str1[i - 1] == str2[j - 1]):
                sub = np.inf
                copy = D[i - 1, j - 1]
            else:
                sub = D[i - 1, j - 1] + 1
                copy = np.inf
            op_arr = [delete, insert, copy, sub]
            D[i, j] = np.min(op_arr)
            op = np.argmin(op_arr)
            if (op == 0):
                # delete, go down
                trace[i, j] = (i - 1, j)
            elif (op == 1):
                # insert, go left
                trace[i, j] = (i, j - 1)
            else:
                # copy or subsitute, go diag
                trace[i, j] = (i - 1, j - 1)
    md = D[n, m]
    return int(md)

@lru_cache(maxsize=200000)
def cosine_similarity(a:str, b:str):
    def parse(pwd, grams=2):
        P = ['[S]'] + [x for x in pwd] + ['[E]']
        ans = defaultdict(int)
        n = len(P)
        for i in range(n-grams+1):
            ch = "".join(P[i:i+grams])
            ans[ch] += 1
        return ans
    A = parse(a)
    B = parse(b)
    p = set()
    for i in A.keys():
        p.add(i)
    for i in B.keys():
        p.add(i)
    p = [x for x in p]
    XX = [A[i]*A[i] for i in p]
    YY = [B[i]*B[i] for i in p]
    XY = [A[i]*B[i] for i in p]
    eps = 10 ** -6
    return sum(XY) / math.sqrt(sum(XX)*sum(YY) + eps)

def rate_count(config, n):
    path = config["path"]
    total_count = 0
    guess_idx = config["guess_idx"]

    dp = [0] * (n+2)

    line_cnt = count_lines(path)

    with open(path, "r") as f:
        for line in tqdm.tqdm(f, desc="lines", total=line_cnt):
            total_count += 1
            line = line.strip("\r\n")
            ss = line.split("\t")
            pwd1 = ss[0]
            pwd2 = ss[1]
            d = edit_distance(pwd1, pwd2)
            value = int(ss[guess_idx])
            if value == -1:
                continue
            if value <= 1000:
                if d <= n:
                    dp[d] += 1
                else:
                    dp[-1] += 1
    
    config["y"] = [x*100/total_count for x in dp][1:]
            
    print("Targuess Counter")
    print(f">>> Total count: {total_count}")

    return config

def save_grouped_bar_chart(line_params, save_path):
    # 设置图形大小
    plt.figure(figsize=line_params["fig_size"])

    data = line_params["line_params"]
    bar_width = 0.35
    gap_width = 0.5
    num_groups = len(data)
    group_width = bar_width * num_groups
    index = np.arange(n+1)*(group_width+gap_width) - (num_groups / 2.0)*bar_width

    # 绘制柱状图
    for i, d in enumerate(data):
        values = d["y"]
        
        plt.bar(index+i*bar_width, values, color=d["color"], edgecolor=d["edgecolor"], hatch=d["hatch"], width=bar_width, align='center')



    # 设置x轴刻度和标签
    plt.xticks(index + (num_groups / 2.0 -0.5)*bar_width, line_params["xticks_val"], fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel(line_params["xlabel"], fontsize=14)
    plt.ylabel(line_params["ylabel"], fontsize=14)

    # 添加图例
    legend_labels = [params['label'] for params in data]
    plt.legend(legend_labels, fontsize=14)
    plt.tight_layout()

    # 保存图形到指定路径
    plt.savefig(save_path)

    print(f">>> Figure saved in {save_path}")

    # 关闭图形
    plt.close()

def main():
    config = demo_config

    for item in config["line_params"]:
        rate_count(item, n=n)
    
    save_grouped_bar_chart(config, config["fig_save"])


main()