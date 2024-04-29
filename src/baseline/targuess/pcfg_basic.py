from util import *
from edit_path import *

def tag_to_str(tag):
    return "".join([str(x) for x in tag])

def str_to_tag(str):
    return (str[0], int(str[1]))

def template_to_str(template):
    return "".join([tag_to_str(x) for x in template])

def generate_template_string(string):
    template = ""
    for char in string:
        if char.isalpha():
            template += 'L'
        elif char.isdigit():
            template += 'D'
        else:
            template += 'S'
    return template

def build_(string):
    if not string:
        return ()
    template = generate_template_string(string)
    merged = []
    count = 1
    for i in range(1, len(template)):
        if string[i] == string[i-1]:
            count += 1
        else:
            merged.append(string[i-1])
            count = 1
    merged.append(string[-1])
    return tuple(merged)

def merge_and_count(pwd, string):
    if not string:
        return ()

    merged = []
    count = 1
    s = pwd[0]
    for i in range(1, len(string)):
        if string[i] == string[i-1]:
            count += 1
            s += pwd[i]
        else:
            merged.append(((string[i-1], count), s))
            count = 1
            s = pwd[i]
    # 处理最后一个字符
    merged.append(((string[-1], count), s))
    # List of (tag, segment)
    return tuple(merged)

def pcfg_decompose(x):
    template = generate_template_string(x)
    return merge_and_count(x, template)

def train_basic_pcfg(pwds):
    model = defaultdict(lambda :defaultdict(int))
    for pwd,cnt in pwds.items():
        items = pcfg_decompose(pwd)
        for tag, segment in items:
            tag = tag_to_str(tag)
            model[tag][segment] += cnt
    res = defaultdict()
    # 只保留topk100
    for k in model:
        total = sum([x for x in model[k].values()])
        ans = [(x, y/total) for x, y in model[k].items()]
        ans = sorted(ans, key=lambda x:-x[1])
        if len(ans)>100:
            ans = ans[:100]
        res[k] = ans
    # 返回pcfg统计结果
    return res

def apply_ht_structure_edit(template, segments, op):
    if op[0] == "hi":
        segments.insert(0, op[2])
        template.insert(0, op[3])
    elif op[0] == "hd":
        segments.pop(0)
        template.pop(0)
    elif op[0] == "ti":
        segments.append(op[2])
        template.append(op[3])
    elif op[0] == "td":
        segments.pop()
        template.pop()
    return segments, template

def apply_ht_edit(tag, segment, op):
    if op[0] == "hi":
        segment.insert(0, op[1])
        tag[1] += 1
    elif op[0] == "hd":
        segment.pop(0)
        tag[1] -= 1
    elif op[0] == "ti":
        segment.append(op[1])
        tag[1] += 1
    elif op[0] == "td":
        segment.pop()
        tag[1] -= 1
    return tag, segment

def generate_pcfg_segment_edit(tag1, segment1, segment2):
    if segment1 == segment2:
        return [(tag_to_str(tag1), "No", None)]
    # segment1 = [x for x in segment1]
    # segment2 = [x for x in segment2]
    ht_edits = generate_ht_edit(segment1, segment2)
    current_segment = [x for x in segment1]
    current_tag = [x for x in tag1]
    edits = []
    for edit in ht_edits:
        prev_tag = tag_to_str(current_tag)
        current_tag, current_segment = apply_ht_edit(current_tag, current_segment, edit)
        edits.append((prev_tag, edit[0], edit[1]))
    return edits

def generate_pcfg_structure_edit(pwd1, pwd2):
    x1 = pcfg_decompose(pwd1)
    x2 = pcfg_decompose(pwd2)
    base1 = [x[0][0] for x in x1]
    base2 = [x[0][0] for x in x2]
    template1 = [tag_to_str(x[0]) for x in x1]
    template2 = [tag_to_str(x[0]) for x in x2]
    segments1 = [x[1] for x in x1]
    segments2 = [x[1] for x in x2]
    ht_edits = generate_ht_structure_edit(base1, base2, segments1, segments2, template1, template2)
    if len(ht_edits) == 0:
        return [("".join(template1), "No", None)], [(x, "No", None) for x in template1]
    edits = []
    current_template = template1
    current_segment = segments1
    for e in ht_edits:
        prev_template = "".join(current_template)
        current_segment, current_template = apply_ht_structure_edit(current_template, current_segment, e)
        edits.append((prev_template, e[0], e[3]))
    assert(len(current_template) == len(template2))
    assert(len(current_segment) == len(segments2))
    n = len(current_template)
    segment_edits = []
    for i in range(n):
        tag = str_to_tag(current_template[i])
        segment_edits.extend(
            generate_pcfg_segment_edit(tag, current_segment[i], segments2[i])
        )
        pass
    return edits, segment_edits
    # return {
    #     "structure_edits":edits, 
    #     "segment_edits": segment_edits
    # }

def main():
    X = [
        "123qwe", 
        "1q2w!",
        "Aq23de$#"
    ]
    for x in X:
        print(x,pcfg_decompose(x))
    
    # pairs = [
    #     ("abc123!", "acn%"),
    #     ("$abc$123!", "1acn%1")
    # ]

    # for pair in pairs:
    #     ans = generate_pcfg_structure_edit(*pair)
    #     print(ans)
    pass


if __name__ == '__main__':
    main()