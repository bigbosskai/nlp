import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
import numpy as np 
import pandas as pd 
import utils

torch.set_default_tensor_type( torch.FloatTensor)

class Config(object):
    word_embed_size = 10
    batch_size = 5 # 每批处理句子的数目
    label_size = 17  # 类别数 y 的纬度
    rnn_hidden_size = 10   # 
    num_words_of_sentence = 113
    lr          = 0.15

class RNN(nn.Module):#here wordsn is not define
    def __init__(self,wordsn):
        super(RNN,self).__init__()
        self.config = Config()
        self.embedding = nn.Embedding(wordsn, self.config.word_embed_size)
        self.rnn = nn.RNN(
                        input_size=self.config.word_embed_size, 
                        hidden_size=self.config.rnn_hidden_size,
                        batch_first=True)
        self.fc = nn.Linear(self.config.rnn_hidden_size, self.config.label_size)

    def forward(self,  x ):
        emout = self.embedding( x )#( bs,num of words of sentence, embedding_dim)
        output,hidden = self.rnn( emout )
        output = output.contiguous().view(-1, self.config.rnn_hidden_size)
        output = self.fc(output)        
        output = output.view(emout.shape[0],emout.shape[1],self.config.label_size)
        return output

def cross_entropy_for_rnn(logit,real_label):
    #logit shape: [bs,n_step, label_size],
    #real_babel shape: [bs,n_step,label_size]
    predict = torch.softmax( logit, dim=2 )
    cross_entropy =  torch.sum( -1.0 *torch.log( predict )*real_label , dim=2)
    return torch.mean(cross_entropy)

def accuracy_and_tag(logit, real_label):
    tag_idx = []
    predict = torch.softmax(logit, dim=2)#bs, n_step,
    value, idx = torch.max(real_label, dim=2)#value:bs, n_step
    effective_label = torch.sum(value, dim=1).tolist()
    total = sum(effective_label)
    hit = 0.
    logt_tag = torch.argmax( predict , dim=2).tolist()
    real_tag = torch.argmax( real_label, dim=2).tolist()

    for effnum, ltag, rtag in zip(effective_label, logt_tag, real_tag):
        hit = hit + sum([1. for t1,t2 in zip(ltag[:int(effnum)],rtag[:int(effnum)]) if t1==t2]   )
        tag_idx.extend(ltag[:int(effnum)])
    print(logt_tag[0])
    print(real_tag[0])
    print(hit)
    return hit/total,tag_idx


data = utils.Data('eng.train.bioes','eng.testa.bioes','vocab.txt')
data.data_to_tensor(cuda=True)


device = torch.device('cuda')
net = RNN(data.wn)
net.to(device)
optimizer = optim.SGD(net.parameters(), net.config.lr)




f=open('predict.txt','w')
acc_list = []
loss_list = []


X_test = data.X_test
Y_test = data.Y_test


def eval(net):
    net.eval()
    #下面测试
    logit = net( X_test )
    loss  = cross_entropy_for_rnn(logit, Y_test)
    acc,tag_idx   = accuracy_and_tag(logit,Y_test)
    tag_name = [data.n2t[idx] for idx in tag_idx]#所有测试数据返回的Tag string
    s = ' '.join(tag_name)
    s = s+'\n'
    f.write( s )
    return acc,loss

# zero epoch
acc,loss = eval(net)
acc_list.append(acc)
loss_list.append(loss)


epoch=500
for i in range(epoch):
    net.train()
    for bx,by in data.next_batch(net.config.batch_size):
        logit = net( bx )
        loss = cross_entropy_for_rnn(logit, by )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc,loss = eval(net)
    acc_list.append( acc )
    loss_list.append( loss )
    print("Iteration:{} loss:{} acc:{}".format(i,loss,acc))

f.close()
f1 = open('acc_and_loss.csv','w')
for a,l in zip(acc_list,loss_list):
    s = str(a)+','+str(l.item())+'\n'
    f1.write(s)
f1.close()