

import tokenizer


t = tokenizer.TransDict()

M = [(k,v) for k,v in t.bag.items()]

print(M[-5:])

print(len(t))
