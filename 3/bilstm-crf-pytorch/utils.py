from torch.utils import data
from torch.utils.data import DataLoader
import torch
import os 
import re
def get_parameter_number(net):
    total_sum = sum(p.numel() for p in net.parameters())
    trainable_sum = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return total_sum,trainable_sum
class CONLL2003(data.Dataset):#
    def __init__(self,root):#./data
        self.train_path = os.path.join(root,'eng.train.bioes')
        self.testa_path = os.path.join(root,'eng.testa.bioes')
        self.testb_path = os.path.join(root,'eng.testb.bioes')
        self.vocab_path = os.path.join(root,'vocab.txt')
        
        self.w2i,self.i2w   = self.gen_w_and_index(self.vocab_path)

        self.Tags =  ['B-LOC','I-LOC','E-LOC','S-LOC','B-ORG','I-ORG','E-ORG','S-ORG','B-PER','I-PER','E-PER','S-PER','B-MISC','I-MISC','E-MISC','S-MISC','O','<START>','<STOP>']
        self.t2i = {v:i for i,v in enumerate(self.Tags)}
        self.i2t = {v:i for i,v in self.t2i.items()}

        self.train = self.read_file(self.train_path)
        self.testa = self.read_file(self.testa_path)
        self.testb = self.read_file(self.testb_path)
        self.train_length = len(self.train)
    
    def gen_w_and_index(self,path):
        with open(path) as fd:
            words = [line.strip() for line in fd]
            num_to_word = dict(enumerate(words))
            word_to_num = {v:k for k,v in num_to_word.items()}
            return  word_to_num, num_to_word
    
    def canonicalize_digits(self,word):
        # 字符串至少有一个字符并且所有字符都是字母则返回 True,否则返回 False
        if word.isalpha(): return word
        word = re.sub("\d", "DG", word)#word='1.23' ==> 'DG.DGDG'
        if word.startswith("DG"):
            word = word.replace(",", "") # remove thousands separator
        return word
    def canonicalize_word(self,word, wordset=None):
        word = word.lower()
        if (wordset != None) and (word in wordset): return word
        # word有可能是数字
        word = self.canonicalize_digits(word) # try to canonicalize numbers
        if (wordset != None) and (word in wordset): return word
        else: return "UUUNKKK" # unknown token

    def read_file(self, path):#return list of list 
        docs = []
        with open(path,'r') as fd:
            curword     = []
            curlabel    = []
            for line in fd:
                # new sentence on -DOCSTART- or blank line
                if re.match(r"-DOCSTART-.+", line) or (len(line.strip()) == 0):
                    if len(curword) > 0 and len(curlabel)>0 :
                        docs.append((curword,curlabel))
                    curword     = []
                    curlabel    = []
                else: # read in tokens
                    tokens = line.strip().split(" ")
                    curword.append( self.canonicalize_word(tokens[0], wordset=self.w2i) )
                    curlabel.append( tokens[1] )

        return docs
    def gen_tensor(self,sentence,labels):
        twords  = torch.tensor([self.w2i[word] for word in sentence], dtype=torch.long)
        tlabels = torch.tensor([self.t2i[label] for label in labels], dtype=torch.long)
        return twords,tlabels

    def __getitem__(self, index):

        words,labels = self.train[index]
        twords  = torch.tensor([self.w2i[word] for word in words], dtype=torch.long)
        tlabels = torch.tensor([self.t2i[label] for label in labels], dtype=torch.long)
        return twords,tlabels

    def __len__(self):
        # pass
        return len(self.train)

# dataset    = CONLL2003('./data')
# dataloader = DataLoader( dataset, batch_size=1, shuffle=False,num_workers=0, drop_last=False)



# dataiter = iter(dataloader)
# sentence,labels = next(dataiter)
# print(sentence.squeeze())
# print(labels.squeeze())