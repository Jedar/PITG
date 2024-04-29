
import argparse
import matplotlib.pyplot as plt
import numpy as np 


# def init_args():
#     cli = argparse.ArgumentParser("Target Guessing Counter")
#     cli.add_argument("-c", "--csv", dest="csv")
#     args = cli.parse_args()
#     return args

dataset = "Collection1"
    
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
            "marker": "*",
            "marker_size": 2,
            "line_width": 1,
            "pwd_idx": 0,
            "guess_idx": 3
        }
        ,
        {
            "label": "PassBERT",
            "path": f"/disk/yjt/PasswordSimilarity/result/targuess/passbert/t_collection1_q_{dataset}_100k_m_passbert_4kw.csv",
            "color": "orange",
            "line_style": "solid",
            "marker": "o",
            "marker_size": 2,
            "line_width": 1,
            "pwd_idx": 0,
            "guess_idx": 3
        },
        {
            "label": "Pass2Path",
            "path": f"/disk/yjt/PasswordSimilarity/result/targuess/pass2path/t_collection1_q_{dataset}_100k_m_pass2path_4kw.csv",
            "color": "red",
            "line_style": "solid",
            "marker": "v",
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
            "marker": ".",
            "marker_size": 2,
            "line_width": 1,
            "pwd_idx": 0,
            "guess_idx": 3
        }
    ],
    "combine":True,
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
    "xlim_low": 1,
    "xticks_val": [
            0,
            10,
            100,
            1000
            
    ],
    "yticks_val": [
            0,
            2,
            4,
            6,
            8,
            10
    ],
    "fig_size": "8 4.95",
    "fig_save": "/disk/clw/target_prompt/data/figure/test32.pdf"

}


def find_med_backtrace(str1, str2, cutoff=-1):
    '''
    This function calculates the Minimum Edit Distance between 2 words using
    Dynamic Programming, and asserts the optimal transition path using backtracing.
    Input parameters: original word, target word
    Output: minimum edit distance, path
    Example: ('password', 'Passw0rd') -> 2.0, [('s', 'P', 0), ('s', '0', 5)]
    '''
    # op_arr_str = ["d", "i", "c", "s"]
    # Definitions:
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
    # print(trace)
    # Find the path of transitions:
    i = n
    j = m
    cursor = trace[i, j]
    md = D[n, m]
    del D, trace
    #print(f'this is md {md}')
    #print(f'this is path{list(reversed(path))}')
    #print(list(reversed(path)))
    return md


def rate_count(config,res_dict):
    #path = args.csv
    # path = '/disk/yjt/PasswordSimilarity/result/csv/4iQ_sample_explainable_test.csv'
    #guess_stone = [1, 10, 100, 1000]
    #guess_stone = list(range(0,1001,10))

    path = config["path"]

    distance_count = [0 for x in range(100)]
    total_count = 0
    
    guess_idx = config["guess_idx"]

    #email_idx = config["email_idx"]
    with open(path, "r") as f:
        for line in f:
            total_count += 1
            line = line.strip("\r\n")
            ss = line.split("\t")
            value = int(ss[guess_idx])

            item_key = (ss[0],ss[1])
            # total_count += 1
            if value == -1:
                continue
                # if value <= 10:
                #     print(ss[0])
            
            ##猜测是100
            # if value >= 100:
            #     continue
            #存最终整合的线
            # if item_key in res_dict.keys():
            #     if res_dict[item_key] > value:
            #         res_dict[item_key] = value
            # else:
            #     res_dict[item_key] = value


            md = find_med_backtrace(ss[0],ss[1])
            distance_count[int(md)] += 1
            
            
    print("Targuess Counter")
    print(f"Total count: {total_count}")
        #for i, value in enumerate(count):
            #print(f"Guess number {guess_stone[i]}: {value / total_count}({value})")
        # print(f"Crack rate: {count / total_count}")
    #res_dict[path_label_dict[path]] = [count,total_count]
    config["distance_count"] = distance_count
    config["total_count"] = total_count
    #print(res_dict[path_label_dict[path]])
    #draw(guess_stone,count,total_count)
    return config,res_dict

# def combine_guess(res_dict):

#     guess_stone = list(range(0,1001,10))
#     count = [0 for x in range(len(guess_stone))]
#     #total_count = 0


#     for key in res_dict.keys():       
#      #   total_count += 1
#         value = res_dict[key]
#         if value == -1:
#             continue
#             # if value <= 10:
#             #     print(ss[0])
#         for i, stone in enumerate(guess_stone):
#             if value < stone:
#                 count[i] += 1
#     combine_dict = {"count":count}
#     return combine_dict

def draw(configs):

    # markers = ['.',',','o','v','s','*','+','x']
    # colors =['b','g','r','c','m','y','k','w']

    x = ["1","2","3","4","5",">5"]
    for line_para in configs["line_params"]:
        y = []
        bigger_than_five = 0
        for i,c in enumerate(line_para["distance_count"]):
            if 0 < i <= 5:
                y.append(c/line_para["total_count"])
            
            else:
                bigger_than_five += c
        y.append(bigger_than_five/line_para["total_count"])

        #y = [c/line_para["total_count"] for c in line_para["count"]]
        plt.plot(x,y,marker = line_para["marker"],color = line_para["color"],label = line_para["label"],linestyle = line_para["line_style"])

    # if configs["combine"]:
    #     y = [c/combine_dict["total_count"] for c in combine_dict["count"]]
    # plt.plot(guess_stone,y,marker = ',',color = "grey",linestyle = 'solid',label = "combine")

    #纵坐标设置
    y = []
    for num in configs["yticks_val"]:
        temp = num/100
        y.append(temp)

    plt.yticks(y, configs["yticks_val"])
    #横坐标设置
    
    #x = configs["xticks_val"]
    #tick_positions = np.linspace(0, 3, 4)
    #plt.xticks(x, [str(val) for val in x])
    #plt.xticks(tick_positions,x)

    #plt.xscale(configs["xscale"])
    plt.yscale(configs["yscale"])

    plt.grid(True)
    plt.xlabel(configs["xlabel"])
    plt.xlabel(configs["ylabel"])
    plt.title("Guess")
    
    
    plt.legend()
    plt.savefig(configs["fig_save"])




def main():
    # args = init_args()
    # path_label_dict = {}
    # paths = ['/disk/yjt/PasswordSimilarity/result/targuess/pass2path/t_collection1_q_4iQ_10k_m_pass2path_4kw.csv',
    #          '/disk/yjt/PasswordSimilarity/result/targuess/passbert/t_collection1_q_4iQ_10k_m_passbert_4kw.csv',
    #          '/disk/yjt/PersonalTarGuess/result/csv/targuessii/t_collection1_4kw_q_4iQ_10k_m_targuessii.csv'
    #          ]
    
    
    # labels = ['pass2path','passbert','targuess']

    # for i,path in enumerate(paths):
    #     path_label_dict[path] = labels[i]
    res_dict = {}
    # #print(path_label_dict)
    for i,config in enumerate(demo_config["line_params"]):
        demo_config["line_params"][i],res_dict = rate_count(config,res_dict)

    #combine_dict = combine_guess(res_dict)
    #combine_dict["total_count"] = demo_config["line_params"][0]["total_count"]
    #res_dict = rate_count(path_label_dict) 
    #guess_stone = list(range(0,1001,10))
    



    #for i,line_config in enumerate(demo_config["line_params"]):
    draw(demo_config)


    pass



if __name__ == '__main__':
    main()
