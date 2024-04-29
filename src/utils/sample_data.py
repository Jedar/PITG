

import random

# 从1亿行数据中随机采样500万行数据
sample_size = 5000000
total_size = 100000000

# 计算采样概率
probability = sample_size / total_size

data_path = '/disk/data/with_email/Collections1_reduce.csv'
output_path = '/disk/yjt/PersonalTarGuess/data/analysis/Collections1_500w.csv'

# 打开输出文件
output_file = open(output_path, 'w')

# 逐行读取数据文件，进行随机采样并写入输出文件
with open(data_path, 'r') as file:
    for line in file:
        if random.random() < probability:
            output_file.write(line)

# 关闭输出文件
output_file.close()

print(f">>> Result saved in : {output_path}")