from collections import defaultdict
import math
import numpy as np

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
    path = []
    while (cursor is not None):
        # 3 possible directions:
        #         print(cursor)
        if (cursor[0] == i - 1 and cursor[1] == j - 1):
            # diagonal - sub or copy
            if (str1[cursor[0]] != str2[cursor[1]]):
                # substitute
                path.append(("s", str2[cursor[1]], cursor[0]))
            i = i - 1
            j = j - 1
        elif (cursor[0] == i and cursor[1] == j - 1):
            # go left - insert
            path.append(("i", str2[cursor[1]], cursor[0]))
            j = j - 1
        else:
            # (cursor[0] == i - 1 and cursor[1] == j )
            # go down - delete
            path.append(("d", None, cursor[0]))
            i = i - 1
        cursor = trace[cursor[0], cursor[1]]
        # print(len(path), cursor)
    md = D[n, m]
    del D, trace
    return md, list(reversed(path))

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

def main():
    w1 = "password"
    w2 = "drowssap"
    print(find_med_backtrace(w1, w2))
    pass

if __name__ == '__main__':
    main()