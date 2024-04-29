from tqdm import tqdm

from collections import defaultdict
import math

SIMILARITY_THRESHOLD = 0.5




def LDS_item_count(source,target):
    #用于生成训练集中各类LDS有多少 返回值为：{L1：{123：1}...etc}
    statistics = {}  # 存储统计结果的字典
    for token in [source,target]:
                structure = ''
                count = 0
                templist = []
                for i,char in enumerate(token):
                    if char.isalpha():
                        if structure == 'L':
                            count += 1
                        else:
                            if structure != '':
                                templist.append({f"{structure}{count}":token[i-count:i]})
                            count = 1
                            structure = 'L'
                    elif char.isdigit():
                        if structure == 'D':
                            count += 1
                        else:
                            if structure != '':
                                templist.append({f"{structure}{count}":token[i-count:i]})
                            count = 1
                            structure = 'D'
                    else:
                        if structure == 'S':
                            count += 1
                        else:
                            if structure != '':
                                templist.append({f"{structure}{count}":token[i-count:i]})
                            count = 1
                            structure = 'S'
                    
                    if i == len(token)-1:
                        if structure != '':
                                templist.append({f"{structure}{count}":token[i-count+1:i+1]})
                
                for item in templist:
                    #print(type(item))
                    for key,value in item.items():
                        #print(key,value)
                        if key not in statistics:
                            statistics[key] = {}

                        if value not in statistics[key]:
                            statistics[key][value] = 1
                        else:
                            statistics[key][value] += 1
    #返回值是一个字典
    return statistics


'''
规则C： NO：0.95 C1：0.01 C2：0.03 C3：0.003 C4：0.007
规则L： NO：0.95 L1：a<->@ 0.02 L2：s<->$ 0.01 L3:o<->0 0.01 L4:i<->1 0.005 L5:0.005

'''
def gain_structure_dict():
    #初始化空的CLSMR字典
    structure_transform_dict = {}
    C_dict = {"No":0,"C1":0,"C2":0,"C3":0,"C4":0.}
    L_dict = {"No":0,"L1":0,"L2":0,"L3":0,"L4":0,"L5":0}
    SM_dict = {"No":0,"Yes":0}
    R_dict = {"No":0,"R1":0,"R2":0}
    for item in [C_dict,L_dict,SM_dict,R_dict]:
        for key,value in item:
            structure_transform_dict[key] = value
    


    return structure_transform_dict



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


def structure_transformer(source,target,structure_transform_dict):
    #更新CLSMR字典
    A = source
    B = target
    edit = []
    d_init = cosine_similarity(A,B)
    dist = d_init
    temp = ''
    for op in structure_transform_dict:
        #used = False
        trans = "No"
        print(op)
        for key in op.keys():
            #print(type(op))
            if len(op) == 5:
                #C
                if key == "C1":
                    temp = A.upper()
                elif key == "C2":
                    temp = A[0].upper() + A[1:] 
                elif key == "C3":
                    temp = A.lower()
                elif key == "C4":
                    temp = A.lower()
                
                if cosine_similarity(temp,B) > dist:
                    trans = key
                    A = temp
                    dist = cosine_similarity(temp,B)
                
            elif len(op )== 6:
                #L
                if key == "L1":
                    temp = A.replace('a','@')
                elif key == "L2":
                    temp = A.replace('s','$')
                elif key == "L3":
                    temp = A.replace('o','0')
                elif key == "L4":
                    temp = A.replace('i','1')
                elif key == "L5":
                    temp = A.replace('e','3')

                if cosine_similarity(temp,B) > dist:
                    trans = key
                    A = temp
                    dist = cosine_similarity(temp,B)
                    
            elif len(op )== 2:
                #SM
                #此处要实现一个将字符串切分为structure的功能，参考LDS_item函数
                if key == "Yes":
                    continue
                if cosine_similarity(temp,B) > dist:
                    trans = key
                    A = temp
                    dist = cosine_similarity(temp,B)

            elif len(op )== 3:
                #R
                if key == "R1":
                    temp = A[0][::-1]
                elif key == "R2":
                    #此处也是一样，要按照structure来reverse
                    continue
                if cosine_similarity(temp,B) > dist:
                    trans = key
                    A = temp
                    dist = cosine_similarity(temp,B)
        edit.append(trans)
    
    if len(edit) < 4:
        for i in range(4 - len(edit)):
            edit.append("No")
    
    if dist >= SIMILARITY_THRESHOLD:
        Cx,Lx,SMx,Rx = edit[0],edit[1],edit[2],edit[3]
        structure_transform_dict[0][Cx] += 1
        structure_transform_dict[1][Lx] += 1
        structure_transform_dict[2][SMx] += 1
        structure_transform_dict[3][Rx] += 1
        yield A
    
            
             

def read_csv(input_path):
    #用于读取文件
    num_file = sum([1 for i in open(input_path,'r')])
    with open(input_path,'r') as f:
            for line in tqdm(f,total = num_file):
                parts = line.strip().split("\t") 
                source = parts[0]
                target = parts[1]
                email = parts[2]


def train()

def main():


if __name__ == '__main__':
    main()