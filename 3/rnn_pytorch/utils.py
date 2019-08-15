import re
from numpy import array
import itertools
import pandas as pd
import numpy as np

def get_parameter_number(net):
    total_sum = sum(p.numel() for p in net.parameters())
    trainable_sum = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_sum,trainable_sum

#config 
# np.random.seed(1234)
class Data(object):
    def __init__(self,trainfilename,testfilename,vocabfile):
        self.raw_docs,self.sentence_max_len = self.load_dataset(trainfilename)
        self.raw_test_docs, _ = self.load_dataset( testfilename )
        self.w2n,self.n2w = self.load_vocabfile(vocabfile)
        self.wn = len(self.w2n)
        tagnames =  ['B-LOC','I-LOC','E-LOC','S-LOC','B-ORG','I-ORG','E-ORG','S-ORG','B-PER','I-PER','E-PER','S-PER','B-MISC','I-MISC','E-MISC','S-MISC','O']
        self.n2t = {i:t for i,t in enumerate(tagnames)}
        self.t2n = {v:k for k,v in self.n2t.items()}
        self.X_train,self.Y_train = self.docs_to_vector('train',self.raw_docs)
        self.X_test ,self.Y_test  = self.docs_to_vector('test',self.raw_test_docs)

    def data_to_tensor(self, cuda=True):
        import torch
        if cuda:
            self.X_train = torch.from_numpy(self.X_train).long().cuda()
            self.Y_train = torch.from_numpy(self.Y_train.astype(np.float32)).cuda()
            self.X_test  = torch.from_numpy(self.X_test).long().cuda()
            self.Y_test  = torch.from_numpy(self.Y_test.astype(np.float32)).cuda()
        else:
            self.X_train = torch.from_numpy(self.X_train).long()
            self.Y_train = torch.from_numpy(self.Y_train.astype(np.float32))
            self.X_test  = torch.from_numpy(self.X_test).long()
            self.Y_test  = torch.from_numpy(self.Y_test.astype(np.float32))

    def next_batch(self,batch_size):
        n_samples = self.X_train.shape[0]
        # ri = np.random.permutation( n_samples )
        # x_train = x_train[ri] # shuffle data
        # y_train = y_train[ri]
        for i in range(0,n_samples,batch_size):
            upper_bound = min(i+batch_size,n_samples)
            ret_x = self.X_train[i:upper_bound]
            ret_y = self.Y_train[i:upper_bound]
            yield ret_x,ret_y
    def load_dataset(self,fname):
        docs = []
        sentence_max_len = 0
        with open(fname) as fd:
            cur = []
            for line in fd:
                # new sentence on -DOCSTART- or blank line
                if re.match(r"-DOCSTART-.+", line) or (len(line.strip()) == 0):
                    if len(cur) > 0:
                        docs.append(cur)
                        if len(cur) > sentence_max_len:
                            sentence_max_len = len(cur)
                    cur = []
                else: # read in tokens
                    cur.append(line.strip().split(" ",1))
                    # cur.append(line.strip().split("\t",1))
            # flush running buffer
            # docs.append(cur)
        return docs,sentence_max_len
    def flatten1(self,lst):
        return list(itertools.chain.from_iterable(lst))

    def canonicalize_digits(self,word):
        if any([c.isalpha() for c in word]): return word
        word = re.sub("\d", "DG", word)
        if word.startswith("DG"):
            word = word.replace(",", "") # remove thousands separator
        return word

    def pad_sequence(self,seq, right=1):
        rl = seq + right*[['pad',""]]

        return [["<s>", ""]] + rl + [["</s>", ""]]
    def canonicalize_word(self,word, wordset=None, digits=True):
        word = word.lower()
        if digits:
            if (wordset != None) and (word in wordset): return word
            word = self.canonicalize_digits(word) # try to canonicalize numbers
        if (wordset == None) or (word in wordset): return word
        else: return "UUUNKKK" # unknown token
    def docs_to_vector(self,ddtype,raw_docs):
        pad =  1#在左边和右边pad上<s> </s>
        if ddtype=='train':
            self.train_sequence_length = [len(seq) for seq in raw_docs ]
        if ddtype=='test':
            self.test_sequence_length  = [len(seq) for seq in raw_docs]
        docs = self.flatten1(  [self.pad_sequence(seq, right=self.sentence_max_len-len(seq)) for seq in raw_docs]  )
        words, tags = zip(*docs)

        words = [self.canonicalize_word(w, self.w2n) for w in words]
        tags = [t.split("|")[0] for t in tags]

        return self.seq_to_windows(words, tags, self.w2n, self.t2n , pad, pad)
    def load_vocabfile(self,vocabfile):
        # wv = loadtxt(wvfile, dtype=float)
        with open(vocabfile) as fd:
            words = [line.strip() for line in fd]
            num_to_word = dict(enumerate(words))
            word_to_num = {v:k for k,v in num_to_word.items()}
        return  word_to_num, num_to_word


    def seq_to_windows(self,words, tags, word_to_num, tag_to_num, left=1, right=1):
        ns = len(words)
        nt = len(tag_to_num)
        X_train = []
        Y_label = []
        for i in range(1, ns, self.sentence_max_len+2):
            x = []
            y = []
            for j in range(i,i+self.sentence_max_len):
                if words[j]=='pad':
                    tagv = [0. for _ in range(nt)]
                else:
                    if tags[j] not in self.t2n:
                        print(j,words[j],tags[j])
                    tagn = self.t2n[tags[j]]
                    tagv = [0. for _ in range(nt)]
                    tagv[tagn] = 1.
                idxs = self.w2n[ words[j] ]
                y.append( tagv )
                x.append( idxs )
            X_train.append( array(x) )
            Y_label.append( array(y) )
        return array(X_train),array(Y_label)




# data = Data('eng.train.bioes', 'vocab.txt')

# print(data.sentence_max_len)

# for bs,ys in data.next_batch(10):
    # print(bs)