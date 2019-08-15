import re
from numpy import array
import itertools
import pandas as pd
import numpy as np


#config 
hidden_size     = 100
hidden_size2    = 100
window_size     = 3
embed_size      = 50
label_size      = 17
lr              = 0.01
l2              = 0.01
dropout         = 0.6

def load_dataset(fname):
    docs = []
    with open(fname) as fd:
        cur = []
        for line in fd:
            # new sentence on -DOCSTART- or blank line
            if re.match(r"-DOCSTART-.+", line) or (len(line.strip()) == 0):
                if len(cur) > 0:
                    docs.append(cur)
                cur = []
            else: # read in tokens
                cur.append(line.strip().split(" ",1))

    return docs

def load_wv(vocabfile):
    # wv = loadtxt(wvfile, dtype=float)
    with open(vocabfile) as fd:
        words = [line.strip() for line in fd]
    num_to_word = dict(enumerate(words))
    word_to_num = {v:k for k,v in num_to_word.items()}
    return  word_to_num, num_to_word

#aid function
def flatten1(lst):
    return list(itertools.chain.from_iterable(lst))

def pad_sequence(seq, left=1, right=1):
    return int(left)*[("<s>", "")] + seq + int(right)*[("</s>", "")]
def canonicalize_digits(word):
    if any([c.isalpha() for c in word]): return word
    word = re.sub("\d", "DG", word)
    if word.startswith("DG"):
        word = word.replace(",", "") # remove thousands separator
    return word
def canonicalize_word(word, wordset=None, digits=True):
    word = word.lower()
    if digits:
        if (wordset != None) and (word in wordset): return word
        word = canonicalize_digits(word) # try to canonicalize numbers
    if (wordset == None) or (word in wordset): return word
    else: return "UUUNKKK" # unknown token

def seq_to_windows(ddtype,words, tags, word_to_num, tag_to_num, left=1, right=1):
    # f = open(ddtype+'.csv','w')
    ns = len(words)
    nt = len(tag_to_num)
    X = []
    y = []
    for i in range(ns):
        if words[i] == "<s>" or words[i] == "</s>":
            continue # skip sentence delimiters
        # save the to file words[i],tags[i]
        # s =words[i]+'@@'+tags[i]+'\n'
        # f.write(s)
        tagn = tag_to_num[tags[i]]
        tagv = [0. for _ in range(nt)]
        tagv[tagn] = 1.
        idxs = [word_to_num[words[ii]] for ii in range(i - left, i + right + 1)]
        X.append(idxs)
        y.append(tagv)
    # f.close()
    return array(X), array(y)

def docs_to_windows(ddtype,docs, word_to_num, tag_to_num, wsize=3):
    pad = int((wsize - 1)/2)#default pad is 1
    docs = flatten1(  [pad_sequence(seq, left=pad, right=pad) for seq in docs]  )



    words, tags = zip(*docs)
    # print(tags[:20])
    words = [canonicalize_word(w, word_to_num) for w in words]
    tags = [t.split("|")[0] for t in tags]
    return seq_to_windows(ddtype,words, tags, word_to_num, tag_to_num, pad, pad)


def increase_postive_samples(X_train,y_train,  tag_to_num):
    #统计
    O_label_num    = np.sum( y_train[:, tag_to_num['O']])
    LOC_label_num  = np.sum( y_train[:, tag_to_num['LOC']])
    MISC_label_num = np.sum( y_train[:, tag_to_num['MISC']])
    ORG_label_num  = np.sum( y_train[:, tag_to_num['ORG']])
    PER_label_num  = np.sum( y_train[:, tag_to_num['PER']])

    inc_loc  = int(O_label_num/LOC_label_num)
    inc_misc = int(O_label_num/MISC_label_num)
    inc_org  = int(O_label_num/ORG_label_num)
    inc_per  = int(O_label_num/PER_label_num)

    Labels   = np.argmax(y_train, axis=1)
    x_list   = [v for v in X_train]
    y_list   = [v for v in y_train]

    X = []
    Y = []
    for x_array,y_array,t in zip(x_list,y_list,Labels):
        if t == tag_to_num['O']:
            X.append(x_array)
            Y.append(y_array)
        elif t == tag_to_num['LOC']:
            for _ in range(inc_loc):
                X.append(x_array)
                Y.append(y_array)
        elif t == tag_to_num['MISC']:
            for _ in range(inc_misc):
                X.append(x_array)
                Y.append(y_array)
        elif t == tag_to_num['ORG']:
            for _ in range(inc_org):
                X.append(x_array)
                Y.append(y_array)
        else:
            for _ in range(inc_per):
                X.append(x_array)
                Y.append(y_array)
    return array(X),array(Y)

