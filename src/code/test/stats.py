import pstats

# 读取 cProfile 输出文件
filename = '/disk/yjt/PersonalTarGuess/profile_2.txt'
stats = pstats.Stats(filename)

# # 打印函数的统计信息
# stats.print_stats()

# # 或者可以使用以下方法来获取统计信息并进行进一步的分析
# # 获取函数的累计执行时间排名前 10 的统计信息
# stats.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)

# 获取函数的执行时间排名前 10 的统计信息
stats.sort_stats(pstats.SortKey.TIME).print_stats(50)

# # 获取函数内部调用关系的统计信息
# stats.print_callers()
# stats.print_callees()