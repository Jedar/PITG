import string
from word2keypress import Keyboard
import json
from ast import literal_eval
import torch

DEFAULT_BAG = list(string.ascii_letters) + list(string.digits) + list(string.punctuation)

DEFAULT_START_TOKEN="[GO]"
DEFAULT_END_TOKEN = "[EOS]"
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_UNK_TOKEN = "[UNK]"

class PasswordTokenizer:
    def __init__(self, vocab=DEFAULT_BAG, 
                 start_token=DEFAULT_START_TOKEN, 
                 end_token=DEFAULT_END_TOKEN, 
                 pad_token=DEFAULT_PAD_TOKEN, 
                 unk_token=DEFAULT_UNK_TOKEN):
        self.bag = [pad_token, start_token, end_token, unk_token] + [x for x in vocab]
        self.dict = {c:i for i,c in enumerate(self.bag)} #vocab to int
        self.dict_inv = {i:c for i,c in enumerate(self.bag)} # int to vocab
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.start_token_id = self.dict[start_token]
        self.end_token_id = self.dict[end_token]
        self.pad_token_id = self.dict[pad_token]
        self.unk_token_id = self.dict[unk_token]
        self.max_len = 16
        pass

    def encode(self, password):
        if len(password) > self.max_len:
            password = password[:self.max_len]
        X = [self.start_token] + [x for x in password] + [self.end_token]
        X = [x if x in self.dict else self.unk_token for x in X]
        return [self.dict[x] for x in X]

    def decode(self, digits, strip=False):
        if strip:
            digits = digits[1:-1]
        X = [self.dict_inv[x] for x in digits]
        return "".join(X)
    
    def __call__(self, pwds, padding=True):
        pwds = [self.encode(pwd) for pwd in pwds]
        if padding:
            return self.padding(pwds)
        return pwds
    
    def padding(self, seqs):
        max_length = max([len(seq) for seq in seqs])
        arr =  [seq + [self.pad_token_id] * (max_length - len(seq)) for seq in seqs]
        return torch.tensor(arr)

    def __len__(self):
        return len(self.bag)

class KBDPasswordTokenizer(PasswordTokenizer):

    def __init__(self, vocab=DEFAULT_BAG, start_token=DEFAULT_START_TOKEN, end_token=DEFAULT_END_TOKEN, pad_token=DEFAULT_PAD_TOKEN, unk_token=DEFAULT_UNK_TOKEN):
        super().__init__(vocab, start_token, end_token, pad_token, unk_token)
        self.kbd = Keyboard()
    
    def encode(self, password):
        pwd = self.kbd.word_to_keyseq(password)
        return super().encode(pwd)

class TransDict:
    def __init__(self, max_len=30, char_bag=DEFAULT_BAG, delete=True, replace=True, insert=True, extra_chars=[" ", "\t", "\x03", "\x04"]):
        self.idx = 0
        self.max_len = max_len
        self.char_bag = [x for x in char_bag] + extra_chars
        self.delete = delete
        self.replace = replace
        self.insert = insert
        self.bag = {
            **self._delete_map(max_len), 
            **self._replace_map(max_len, self.char_bag), 
            **self._insert_map(max_len, self.char_bag)
            }
        self.str_bag = {str(k):v for k,v in self.bag.items()}
        pass

    def _delete_map(self, max_len):
        bag = {}
        if not self.delete:
            return bag
        for i in range(max_len):
            key = ("d", None, i)
            bag[key] = self.idx
            self.idx += 1
        return bag
    
    def _insert_map(self, max_len, char_bag):
        bag = {}
        if not self.insert:
            return bag
        for c in char_bag:
            for i in range(max_len):
                key = ("i", c, i)
                bag[key] = self.idx
                self.idx += 1
        return bag
    
    def _replace_map(self, max_len, char_bag):
        bag = {}
        if not self.replace:
            return bag
        for c in char_bag:
            for i in range(max_len):
                key = ("s", c, i)
                bag[key] = self.idx
                self.idx += 1
        return bag
    
    def __len__(self):
        return self.idx


def path2word(word, path):
    '''
    This function decodes the word in which the given path transitions the input word into.
    This is the KeyPress version, which handles the keyboard representations.
    If one of the parts components is not feasible (e.g removing a char from out of range index), it skips it
    Input parameters: original word, transition path
    Output: decoded word
    '''
    final_word = []
    word_len = len(word)
    path_len = len(path)
    i = 0
    j = 0
    while (i < word_len or j < path_len):
        if ((j < path_len and path[j][2] == i) or (i >= word_len and path[j][2] >= i)):
            if (path[j][0] == "s"):
                # substitute
                final_word.append(path[j][1])
                i += 1
                j += 1
            elif (path[j][0] == "d"):
                # delete
                i += 1
                j += 1
            else:
                # "i", insert
                final_word.append(path[j][1])
                j += 1
        else:
            if (i < word_len):
                final_word.append(word[i])
                i += 1
            if (j < path_len and i > path[j][2]):
                j += 1
    return "".join(final_word)

class TransTokenizer:
    def __init__(self, trans_dict=TransDict(), start_token=DEFAULT_START_TOKEN, end_token=DEFAULT_END_TOKEN, pad_token=DEFAULT_PAD_TOKEN, unk_token=DEFAULT_UNK_TOKEN):
        TRANS_to_IDX = trans_dict.bag
        IDX_to_TRANS = {v: k for k, v in TRANS_to_IDX.items()}
        self.bag = [pad_token, start_token, end_token, unk_token] + [x for x in IDX_to_TRANS.values() ]
        
        #记录
        self.dict = {c:i for i,c in enumerate(self.bag)}
        self.dict_inv = {i:c for i,c in enumerate(self.bag)}
        self.start_token = start_token
        self.end_token = end_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.start_token_id = self.dict[start_token]
        self.end_token_id = self.dict[end_token]
        self.pad_token_id = self.dict[pad_token]
        self.unk_token_id = self.dict[unk_token]
        self.extra_token_ids = [
            self.start_token_id, self.end_token_id, self.pad_token_id, self.unk_token_id
        ]
        self.str_bag = {str(k):v for k,v in self.dict.items()}

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.str_bag, f)

    def get_encode_dict(self):
        return self.dict
    
    def get_decode_dict(self):
        return self.dict_inv
    
    def encode(self, seq):
        X = [x for x in seq]
        return [self.dict[x] for x in X]
    
    def decode(self, pwd, digits):
        X = [self.dict_inv[x] for x in digits if x not in self.extra_token_ids]
        return path2word(pwd, X)
    
    def __len__(self):
        return len(self.bag)

class EntityTokenizer:
    def __init__(self, entities, default_token="<SOS>"):

        self.entities = entities

        self.bag = [default_token] + entities
        
        #记录
        self.dict = {c:i for i,c in enumerate(self.bag)}
        self.dict_inv = {i:c for i,c in enumerate(self.bag)}
        self.default_token = default_token
        self.default_token_id = self.dict[default_token]
        self.str_bag = {str(k):v for k,v in self.dict.items()}

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.str_bag, f)

    def get_encode_dict(self):
        return self.dict
    
    def get_decode_dict(self):
        return self.dict_inv
    
    def default_values(self):
        return [self.default_token_id]
    
    def __call__(self, items):
        return torch.tensor([[self.encode(x)] for x in items])

    def encode(self, entity):
        if entity not in self.dict:
            entity = self.default_token
        return [self.dict[entity]]
    
    def decode(self, items):
        raise NotImplementedError()
    
    def __len__(self):
        return len(self.bag)

class RegionTokenizer(EntityTokenizer):
    regions = ['.com', '.ru', '.fr', '.uk', '.net', '.it', '.de', 'unk', '.jp', '.pl', '.cz', '.es', '.ca', '.br', '.au', '.tw', '.hu', '.in', '.nl', '.edu', '.couk', '.se', '.cn', '.ua', '.org', '.at', '.hk', '.dk', '.gr', '.be', '.sk', '.id', '.bg', '.ch', '.lv', '.combr', '.eu', '.mx', '.pt', '.ar', '.ph', '.comtw', '.sg', '.comcn', '.ie', '.comau', '.no', '.my', '.vn', '.fm', '.za', '.nz', '.us', '.comhk', '.ee', '.commx', '.by', '.cojp', '.kr', '.fi', '.comph', '.mil', '.info', '.lt', '.comar', '.coin', '.co', '.compt', '.comsg', '.il', '.coid', '.hr', '.cl', '.tr', '.commy', '.gov', '.th', '.ro', '.cc', '.comvn', '.biz', '.az', '.tv', '.ae', '.nu', '.tn', '.is', '.top', '.ws', '.ry', '.con', '.rs', '.me', '.sa', '.so', '.su', '.pk', '.jp_', '.si', '.lu']

    def __init__(self, default_token="<SOS>"):
        super().__init__(self.regions, default_token)

class HostTokenizer(EntityTokenizer):
    hosts = ['yahoo', 'hotmail', 'mail', 'unk', 'aol', 'live', 'bk', 'list', 'msn', 'comcast', 'inbox', 'web', 'gmx', 'sbcglobal', 'yandex', 'libero', 'wp', 'rambler', 'cox', 'seznam', '163', 'verizon', 'o2', 'free', 'interia', 'tiscali', 'wanadoo', 'orange', 'alice', 'charter', 't-online', 'aim', 'btinternet', '126', 'outlook', 'att', 'op', 'naver', 'bellsouth', 'juno', 'qip', 'virgilio', 'ya', 'ntlworld', 'centrum', 'icloud', 'laposte', 'ukr', 'blueyonder.co', 'sina', 'me', 'abv', 'mac', 'earthlink', 'qq', 'tlen', 'onet', 'arcor', 'vp', 'netzero', 'sky', 'online', 'lycos', 'bol.com', 'tin', 'neuf', 'bigpond', 'poczta', 'mchsi', 'bigmir', 'shaw', 'nate', 'sfr', 'netscape', 'azet', 'bluewin', 'hot']

    def __init__(self, default_token="<SOS>"):
        super().__init__(self.hosts, default_token)

if __name__ == '__main__':
    path = '/disk/yjt/PersonalTarGuess/src/code/config/pitg_trans.json'
    d = TransTokenizer()
    # d.to_json(path)

    t3 = RegionTokenizer()
    print(len(t3))
    print(t3.encode(None))