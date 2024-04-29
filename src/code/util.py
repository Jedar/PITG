import re

def parse_email(email, default_token="<SOS>"):
    pattern = r'^([^@]+)@([^@]+)(\.[^@]+)$'
    matches = re.match(pattern, email)
    if matches:
        username = matches.group(1)
        domain = matches.group(2)
        top_level_domain = matches.group(3)
        return username, domain.lower(), top_level_domain.lower()
    else:
        return default_token, default_token, default_token
    
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

def read_query(file_path):
    pwds = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip("\r\n").split("\t")
            src = line[0]
            target = line[1]
            email = line[2]
            pwds.append((src, target, email))
    return pwds

