import argparse
from typing import TextIO
from tqdm import tqdm
import sys
from collections import defaultdict
import itertools
import string

def find_keyboard_row_column(char):
    # I'm leaving off '`' but it is rarely used in keyboard combos and
    # it makes the code cleaner
    row1 = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', '=']
    s_row1 = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_', '+']

    row2 = ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p', '[', ']', '\\']
    s_row2 = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P', '{', '}', '|']

    row3 = ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', ';', '\'']
    s_row3 = ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L', ':', '"']

    row4 = ['z', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '/']
    s_row4 = ['Z', 'X', 'C', 'V', 'B', 'N', 'M', '<', '>', '?']

    if char in row1:
        return 0, row1.index(char)

    if char in s_row1:
        return 0, s_row1.index(char)

    if char in row2:
        return 1, row2.index(char)

    if char in s_row2:
        return 1, s_row2.index(char)

    if char in row3:
        return 2, row3.index(char)

    if char in s_row3:
        return 2, s_row3.index(char)

    if char in row4:
        return 3, row4.index(char)

    if char in s_row4:
        return 3, s_row4.index(char)

    # Default value for keys that are not checked + non-ASCII chars
    return None


# Finds if a new key is next to the previous key
#
def is_next_on_keyboard(past, current):
    # Check to see if either past or current keys are not valid
    if (past is None) or (current is None):
        return False

    # Adding exclusion for repeated characters, aka '112233'
    if (current[0] == past[0]) and (current[1] == past[1]):
        return True

    # If they both occur on the same row (easy)
    if current[0] == past[0]:
        if (current[1] == past[1]) or (current[1] == past[1] - 1) or (current[1] == past[1] + 1):
            return True

    # If it occurs one row down from the past combo
    #
    # Gets a bit weird since they "rows" don't exactly line up
    # aka 'w' (pos 1) is next to 'a' (pos 0) and 's' (pos 1)
    #
    elif current[0] == past[0] + 1:
        if (current[1] == past[1]) or (current[1] == past[1] - 1) or (current[1] == past[1] + 1):
            return True

    # If it occurs one row up from the past combo
    elif current[0] == past[0] - 1:
        if (current[1] == past[1]) or (current[1] == past[1] + 1) or (current[1] == past[1] - 1):
            return True

    # The two keys are not adjacent according to the checked keyboard
    return False


# Filters keyboard walks to try and limit false positives
#
# Currently only defining "interesting" keyboard combos as a combo that has
# multiple types of characters, aka alpha + digit
#
# Also added some filters for common words that tend to look like
# keyboard combos
#
# Note, this can cause some false negatives in certain cases where a true
# keyboard combo will happen after a false positive check but still be
# part of the original como. For example 'er5tgb'
#
# Haven't seen this much in user behavior so it shouldn't have much impact
# but want to disclose that for future coders. May eventually want to add
# checks for that.
#
def interesting_keyboard(combo):
    # Length check
    if len(combo) < 3:
        return False

    # Filter combos that start with "likely" partial words
    #
    # These occur from common english words that look like keyboard combos
    # E.g. 'deer43'
    #
    if combo[0] == 'e':
        return False

    if (combo[1] == 'e') and (combo[2] == 'r'):
        return False

    if (combo[0] == 't') and (combo[1] == 'y'):
        return False

    if (combo[0] == 't') and (combo[1] == 't') and (combo[2] == 'y'):
        return False

    if combo[0] == 'y':
        return False

    # Reject words that look like keyboard combos
    #
    # Eventually might want to read in a blacklist from a file vs
    # hardcoding it here
    #
    # TODO: Some of the shorter strings will also cover the longer strings
    #       That is a function of how I'm currently adding into the blacklist
    #       May want to clean this up a bit, though it is useful to record
    #       for future research.
    #
    false_positive_words = [
        "drew",
        "kiki",
        "fred",
        "were",
        "pop",
    ]
    full_lower_word = ''.join(combo).lower()

    for item in false_positive_words:
        if item in full_lower_word:
            return False

    # Check for complexity requirements
    alpha = 0
    special = 0
    digit = 0
    for c in combo:
        if c.isalpha():
            alpha = 1
        elif c.isdigit():
            digit = 1
        else:
            special = 1

    # If it meets all the complexity requirements
    if (alpha + special + digit) >= 1:
        return True

    return False


# Looks for keyboard combinations in the training data for a section
#
# For example 1qaz or xsw2
#
# Variables:
#
#     password: The current section of the password to process
#               When first called this will be the whole process.
#               This function calls itself recursively so will then parse
#               smaller chunks of this password as it goes along
#
#     min_keyboard_run: The minimum size of a keyboard run
#
# Returns:
#    There are two return values:
#
#    section_list, found_list
#
#    section_list: A list of the sections to return
#                   E.g. input password is 'test1qaztest'
#                   section_list should return:
#                      [('test',None),('1qaz','K4'),('test',None)]
#
#    found_list: A list of every keyboard combo found when parsing password
#
def detect_keyboard_walk(password, min_keyboard_run=3):
    # The keyboard position of the last key processed
    past_pos = None

    # The current keyboard combo
    cur_combo = []

    # The current found list
    found_list = []

    # The current section list of parsing
    section_list = []

    # Loop through each character to find the combos
    for index, x in enumerate(password):

        # Find the current location of the key on the keyboard
        pos = find_keyboard_row_column(x)

        # Check to see if a run is occuring, (two keys next to each other)
        is_run = is_next_on_keyboard(past_pos, pos)

        # If it is a run, keep it going!
        if is_run:
            cur_combo.append(x)

        # The keyboard run has stopped
        else:
            if len(cur_combo) >= min_keyboard_run:

                # Look at saving this keyboard combo
                #
                # See if the keyboard combo is interesting enough to save
                if interesting_keyboard(cur_combo):

                    # Save the results
                    found_list.append(''.join(cur_combo))

                    # Update base structure mask
                    #
                    # Update any unprocessed sections before the current run
                    if len(cur_combo) != index:
                        section_list.append((password[0:index - len(cur_combo)], None))

                    # Update the mask for the current run
                    section_list.append((''.join(cur_combo), "K" + str(len(cur_combo))))

                    # If not the last section, go recursive and call it with
                    # what's remaining
                    if index != (len(password)):
                        temp_section, temp_found = detect_keyboard_walk(password[index:])

                        # update info if needed
                        section_list.extend(temp_section)
                        if temp_found:
                            found_list.extend(temp_found)

                        # Now return since we don't want to parse the same data twice
                        return section_list, found_list

            # Start a new run
            cur_combo = [x]

        # What was new is now old. Update the previous position
        past_pos = pos

    # Update the last run if needed
    if len(cur_combo) >= min_keyboard_run:

        # Look at saving this keyboard combo
        #
        # See if the keyboard combo is interesting enough to save
        if interesting_keyboard(cur_combo):

            # Save the results
            found_list.append(''.join(cur_combo))

            # Update base structure mask
            #
            # Update any unprocessed sections before the current run
            if len(cur_combo) != len(password):
                section_list.append((password[0:len(password) - len(cur_combo)], None))

            # Update the mask for the current run
            section_list.append((''.join(cur_combo), "K" + str(len(cur_combo))))

        # Not treating it as a keyboard combo since it is not intersting
        else:
            section_list.append((password, None))

    # No keyboard run found
    else:
        section_list.append((password, None))

    return section_list, found_list

#qwer -> 1234/1qza  qwe->qaz/qwer  asd -> asdf/wsx  wsx -> 2wsx/wer  asdf->1234/zxcv  5678-> qwer,1234  qa -> qwe/qaz

def extend_sequence(markov_dict, pattern):
    if pattern in markov_dict:
        next_chars = markov_dict[pattern]
        return pattern + next_chars[0]  # 返回第一个下一个字符
    else:
        return None

def remove_substrings(pwd, substrings):
    for substring in substrings:
        pwd = pwd.replace(substring, " ")
    result = [word for word in pwd.split(" ") if word]
    return result

SEQUENCE_DICTIONARY = {'qwer':['1234','1qza'],'qwe': ['qaz','qwer'] , 'asd':['asdf','wsx'] ,'wsx' :['2wxs','wer'],'asdf':['1234','zxcv'],'5678':['qwer','1234'],'qa':['qwe','qaz'] }

def find_patterns(pwd):
    patterns = ["qwertyuiop[]","asdfghjkl;'","zxcvbnm,./","","][poiuytrewq","';lkjhgfdsa","/.,mnbvcxz","abcdefghijklmnopqrstuvwxyz",
                "ABCDEFGHIJKLMNOPQRSTUVWXYZ","ZYXWVUTSRQPONMLKJIHGFEDCBA","zyxwvutsrqponmlkjihgfedcba"]
    substrings = []
    current_substring = ""

    for pattern in patterns:
        for char in pwd:
            
            if char in pattern:
                if not current_substring:
                    current_substring = char
                elif current_substring[-1] in pattern and pattern.index(char) == pattern.index(current_substring[-1]) + 1:
                    current_substring += char
                else:
                    if len(current_substring) > 2:
                        substrings.append(current_substring)
                    current_substring = char
            else:
                if len(current_substring) > 2:
                    substrings.append(current_substring)
                current_substring = ""
        for substring in substrings:
            pwd = pwd.replace(substring,"")
        

    if len(current_substring) > 2:
        substrings.append(current_substring)
    return substrings


def character_sequence(markov_dict,pwds):
    #_,keyboard_tokens = detect_keyboard_walk(pwds[0])
    tokens = find_patterns(pwds[0])
    
    other_tokens = remove_substrings(pwds[0],tokens)
    #print(tokens)
    #print(other_tokens)
    tokens.extend(other_tokens)
    #print(tokens)
    if len(tokens) > 0:
        
        for i,token in enumerate(tokens):
            change_list = [token]
            extends = extend_sequence(markov_dict,token)
            if extends is not None:
                change_list.append(extends)
            if token in SEQUENCE_DICTIONARY:
                for item in SEQUENCE_DICTIONARY[token]:
                    change_list.append(item)
           
            for item in change_list:
                temp_token = [x for x in tokens]
                temp_token[i] = item
                permutations = itertools.permutations(temp_token)
                for perm in permutations:
                    target_candidate = ''.join(perm)
                    #print(target_candidate)
                    yield target_candidate         
    


def deletion(pwds,min_len = 6):
    #只有在字符数量超过6时才应用
    pwd = pwds[0]
    for char in pwd:
        if len(pwd) < 6:
            break
        if char.isdigit():
            pwd = pwd.replace(char, "", 1)
            #print(pwd)
            yield pwd
                    
    for char in pwd:
        if len(pwd) < 6:
            break
        if char in string.punctuation:
            pwd = pwd.replace(char, "", 1)
            #print(pwd)
            yield pwd

    for char in pwd:
        if len(pwd) < 6:
            break
        if char.isupper():
            pwd =pwd.replace(char, "", 1)
            #print(pwd)
            yield pwd
        
    for char in pwd:
        if len(pwd) < 6:
            break
        if char.islower():
            #print(pwd)
            pwd = pwd.replace(char, "", 1)
            yield pwd
    


    for i in range(len(pwds[0])):
        if len(pwds[0][i+1:]) < 6:
            break
        new_string = pwds[0][i+1:]
        yield new_string

    for i in range(len(pwds[0])-1, -1, -1):
        if len(pwds[0][:i]) < 6:
            break
        new_string = pwds[0][:i]
        yield new_string

    start = 0
    end = len(pwds[0])

    mode = 0
    while start < end:
        if len(pwds[0][start+1:end]) < 6:
            break
        new_string = pwds[0][start+1:end]
        yield new_string
        
        if mode % 2 == 1:
            start += 1
            mode+=1
        else:
            end -= 1
            mode+=1


def insertion(pwds):
    missing_groups = []
    if not any(char.isdigit() for char in pwds[0]):
        missing_groups.append("Digit")
    if not any(char in string.punctuation for char in pwds[0]):
        missing_groups.append("Symbol")
    if not any(char.isupper() for char in pwds[0]):
        missing_groups.append("Uppercase letter")
    if not any(char.islower() for char in pwds[0]):
        missing_groups.append("Lowercase letter")

    for group in missing_groups:
        if group == "Digit":
            for digit in string.digits:
                yield digit + pwds[0]
                yield pwds[0] + digit
        elif group == "Symbol":
            for symbol in string.punctuation:
                yield symbol + pwds[0]
                yield pwds[0] + symbol
        elif group == "Uppercase letter":
            for uppercase_letter in string.ascii_uppercase:
                yield uppercase_letter + pwds[0]
                yield pwds[0] + uppercase_letter
        elif group == "Lowercase letter":
            for lowercase_letter in string.ascii_lowercase:
                yield lowercase_letter + pwds[0]
                yield pwds[0] + lowercase_letter
    
    grams = ['08','01','07','23','06','09','12','05','21','04','11','22','02','13','03','69','00','10','88','20'
             ,'123','087','007','083','084','089','086','666','085','man','143','boy','321','101','420','456','000'
             ,'001','777','ita']

    for item in grams:
        yield item + pwds[0]
        yield pwds[0] + item

def capitalization(pwds):
    
    yield pwds[0].upper()
    for i in range(len(pwds[0])):
        yield pwds[0][:i+1].upper() + pwds[0][i+1:]

    for i in range(len(pwds[0]), -1, -1):
        yield pwds[0][:i] + pwds[0][i:].upper()

    start = 0
    end = len(pwds[0])
    mode = 0
    while start < end:
        #ew_string = pwds[0][start+1:end]
        yield pwds[0][:start+1].upper() +pwds[0][start+1:end] +pwds[0][end:].upper()
        if mode % 2 == 1:
            start += 1
            mode+=1
        else:
            end -= 1
            mode+=1

def reversals(pwds):
    yield (pwds[0][::-1])

def leet_speak(pwds):
    speak_dict = {'a':'@','o':'0','a':'@','s':'$','i':'1','e':'3','t':'7'}
    # 逐一修改的版本
    # for i,char in enumerate(pwds[0]):
    #     if char in speak_dict:
    #         temp = pwds[0][:i]+ speak_dict[char]+ pwds[0][i+1:]
    #         yield temp
    for key in speak_dict:
        yield pwds[0].replace(key,speak_dict[key])
    
    wholy_dict = {'A':['/-\\','/\\','4','@'],'B':['|3','8','|o'],'C':['(','<','K','S'],'D':['|)','o|','|>','<|'],'E':['3'],
                  'F':['|=','ph'],'G':['(','9','6'],'H':['|-|',']-[','}-{','(-)',')-(','#'],'I':['l','1','|','!',']['],
                  'J':['_|'], 'K': ['|<','/<','\\<','|{'],'L':['|_','|','1'] , 'M':['|\\/|','/\\/\\','(\\/)',' /\\\\','/|\\','/v\\']
                  ,'N':['|\\|',' /\\/','|\\\\|',' /|/'],'O':['0','()','[]','\{\}'],'P':['|2','|D'],'Q':['(,)','kw'],'R':['|2','|Z','|?'],
                  'S':['5','$'],'T':['+','\'][\'','7'],'U':['|_|'],'V':['|/','\\|','\\/','/'],'W':['\\/\\/','\\|\\|','|/|/','\\|/','\\^/','//'],
                  'X':['><','}{'],'Y':['`/','\'/','j'],'Z':['2','(\)']
                  }

    for key in wholy_dict:
        for value in wholy_dict[key]:
            yield pwds[0].replace(key,value)


def split_string_by_type(string):
    result = []
    current_type = None
    current_word = ''
    
    for char in string:
        if char.isupper():
            new_type = 'upper'
        elif char.islower():
            new_type = 'lower'
        elif char.isdigit():
            new_type = 'digit'
        else:
            new_type = 'special'
        
        if new_type != current_type:
            if current_word:
                result.append(current_word)
            current_type = new_type
            current_word = ''
        
        current_word += char
    
    if current_word:
        result.append(current_word)
    
    return result

def substring_movement(pwds):
    groups = split_string_by_type(pwds[0])
    group_permutations = itertools.permutations(groups)
    
    cnt = 0
    for permutation in group_permutations:
        new_string = "".join(permutation)
        cnt += 1
        if cnt > 10:
            break
        yield new_string

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end_of_word = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        current = self.root
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
        current.is_end_of_word = True

def read_dictionary(filename):
    dictionary = []
    with open(filename, 'r') as file:
        for line in file:
            word = line.strip()
            dictionary.append(word)
    return dictionary

def split_string(dictionary, pwds,trie):
    result = []
    current = ""
    i = 0
    while i < len(pwds[0]):
        node = trie.root
        j = i
        while j < len(pwds[0]) and pwds[0][j] in node.children:
            node = node.children[pwds[0][j]]
            if node.is_end_of_word:
                current += pwds[0][j]
                result.append(current)
                current = ""
            else:
                current += pwds[0][j]
            j += 1

        if current:
            result.append(current)
            current = ""

        if j == i:
            result.append(pwds[0][i])
            i += 1
        else:
            i = j
    # 首字母大写
    for i in range(len(result)):
        if result[i] in dictionary:
            result[i] = result[i].capitalize()
    yield ''.join(result)

            

def generate_markov_dict(order):
    string_list = ["1234567890", "asdfghjkl;'", "qwertyuiop[]","zxcvbnm,./","abcdefghijklmn"]
    markov_dict = {}
    for string in string_list:
        for i in range(len(string) - order):
            key = string[i:i+order]
            value = string[i+order]
            if key in markov_dict:
                markov_dict[key].append(value)
            else:
                markov_dict[key] = [value]
    return markov_dict

PASSWORD_SET = set()

def guess(source,target,trie,max_guess=1000,wiki_dictionary=[],markov_dict = []):
    pwds = [source,target]

    char_seq = character_sequence(markov_dict,pwds)
    delete = deletion(pwds)
    insert = insertion(pwds)
    capital = capitalization(pwds)
    rever = reversals(pwds)
    leet = leet_speak(pwds)
    submovement = substring_movement(pwds)
    subwordcapi = split_string(wiki_dictionary, pwds,trie)
    #generaters = [capital,rever,leet,submovement,subwordcapi]
    generaters = [char_seq,delete,insert,capital,rever,leet,submovement,subwordcapi]
    cracked = False
    count = 0
    for generator in generaters:
        for pwd in generator:
            # print(type(pwd))
            # print(pwd)
            if pwd in PASSWORD_SET:
                continue
            else:
                PASSWORD_SET.add(pwd)
            if pwd == pwds[1]:
                return count
            count += 1
            if count > max_guess:
                return -1
        if cracked:
            return count
    return -1 

def main():
    input_path = '/disk/data/targuess/3_query/Collection1_100k.csv'
    output_path = '/disk/clw/target_prompt/result/das/Collection1_100k_das.csv'
    dictionary_file = '/disk/clw/target_prompt/data/wiki.txt' 
    #所有的树和字典都作为外部变量传
    wiki_dictionary = read_dictionary(dictionary_file)
    markov_dict  = generate_markov_dict(4)
    trie = Trie()
    for word in wiki_dictionary:
        trie.insert(word)

    num_file = sum([1 for i in open(input_path,'r')])
    with open(input_path,'r') as f:
        with open(output_path,'w') as output:
            for line in tqdm(f,total = num_file):
                parts = line.strip().split("\t") 
                source = parts[0]
                target = parts[1]
                email = parts[2]
                gn = guess(source, target, max_guess=1000,trie=trie,wiki_dictionary=wiki_dictionary,markov_dict=markov_dict)
                output.write(f"{source}\t{target}\t{email}\t{gn}\n")


if __name__ == '__main__':
    main()
