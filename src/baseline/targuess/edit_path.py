from collections import defaultdict
import math

def longest_common_substring(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]  # 初始化动态规划表

    max_len = 0  # 最长公共子串的长度
    end_index = 0  # 最长公共子串的结束位置

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_index = i  # 更新最长公共子串的结束位置
            else:
                dp[i][j] = 0

    start_index = end_index - max_len  # 计算最长公共子串的起始位置
    longest_common_substring = s1[start_index:end_index]  # 提取最长公共子串

    return longest_common_substring

def generate_ht_edit(source, target):
    old_src = source
    old_target = target
    lcs = longest_common_substring(source, target)
    if len(lcs) == 0:
        return [('hd', None) for _ in source] + [('ti', x) for x in target]
    n = len(lcs)
    # 从 source 到 lcs 的删除操作
    delete_ops = []
    while source != lcs:
        op = False
        if source[:n] != lcs:
            source = source[1:]
            delete_ops.append(('hd', None))
            op = True
        elif source[-n:] != lcs:
            source = source[:-1]
            delete_ops.append(('td', None))
            op = True
        if not op and source != lcs:
            source = source[1:]
            delete_ops.append(('hd', None))
    # # 从 lcs 到 target 的插入操作
    insert_ops = []
    while lcs != target:
        if target[:n] != lcs:
            insert_ops.append(('hi', target[0]))
            target = target[1:]
        else:
            insert_ops.append(('ti', target[-1]))
            target = target[:-1]
        if len(target) == 0:
            raise RuntimeError(f"Error when generate edit for: {old_src} {old_target} {lcs}")
    return delete_ops + insert_ops[::-1]

def generate_ht_structure_edit(source, target, pwd1, pwd2, template1, template2):
    '''
    source, target are password basic structures like 'LDS'.
    pwd1, pwd2 are password segments line ['123', 'abc].
    '''
    lcs = longest_common_substring(source, target)
    n = len(lcs)
    # 从 source 到 lcs 的删除操作
    delete_ops = []
    while source != lcs:
        op = False
        if source[:n] != lcs:
            source = source[1:]
            delete_ops.append(('hd', None, None, None))
            op = True
        elif source[-n:] != lcs:
            source = source[:-1]
            delete_ops.append(('td', None, None, None))
            op = True
        if not op and source != lcs:
            source = source[1:]
            delete_ops.append(('hd', None, None, None))
    # # 从 lcs 到 target 的插入操作
    insert_ops = []
    while lcs != target:
        if target[:n] != lcs:
            insert_ops.append(('hi', target[0], pwd2[0], template2[0]))
            target = target[1:]
            pwd2 = pwd2[1:]
            template2 = template2[1:]
        else:
            insert_ops.append(('ti', target[-1], pwd2[-1], template2[-1]))
            target = target[:-1]
            pwd2 = pwd2[:-1]
            template2 = template2[:-1]
        if len(target) == 0:
            raise RuntimeError(f"Error when generate edit for: {pwd1}, {pwd2}")
    return delete_ops + insert_ops[::-1]

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