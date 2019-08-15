import torch
import torch.nn as nn
import torch.optim as optim 
import torch.nn.functional as F 
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader  
import sys
import numpy as np 
import pandas as pd 
import utils

torch.set_default_tensor_type( torch.FloatTensor)

criterion = nn.CrossEntropyLoss()
class Config(object):
    word_embed_size = 10
    batch_size = 5 # 每批处理句子的数目
    label_size = 17  # 类别数 y 的纬度
    lstm_hidden_size = 20   # 
    num_words_of_sentence = 113
    lr          = 0.2


class Lstm(nn.Module):
    def __init__(self,wordsn):
        super(Lstm,self).__init__()
        self.config = Config()
        self.embedding = nn.Embedding(wordsn, self.config.word_embed_size)
        self.lstm = nn.LSTM(
                        input_size=self.config.word_embed_size, 
                        hidden_size=self.config.lstm_hidden_size,
                        batch_first=True)
        self.fc = nn.Linear(self.config.lstm_hidden_size, self.config.label_size)

    def forward(self,  x ):
        emout = self.embedding( x )#( bs,num of words of sentence, embedding_dim)
        output,(hidden,cell) = self.lstm( emout )
        output = output.contiguous().view(-1, self.config.lstm_hidden_size)
        logit = self.fc(output)
        logit = logit.view(x.shape[0],x.shape[1],self.config.label_size)
        return logit

def cross_entropy_for_rnn(logit,real_label):
    #logit shape: [1 ,8 , 17],
    #real_babel shape: [1,8]
    logit = logit.squeeze(0)
    real_label = real_label.squeeze(0)
    return criterion(logit, real_label)


dataset    = utils.CONLL2003('./data')
dataloader = DataLoader( dataset, batch_size=1, shuffle=False,num_workers=0, drop_last=False)
net = Lstm(dataset.wn)
optimizer = optim.SGD(net.parameters(), net.config.lr)
scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.96)

test = dataset.testa

total_sum,trainable_sum = utils.get_parameter_number(net)
print("参数总数,训练参数")
print(total_sum,trainable_sum)

f=open('predict.txt','w')
acc_list = []
loss_list = []


def eval(net):
    net.eval()
    #下面测试
    taglist = []
    hit = 0
    a = 0
    loss = 0.
    with torch.no_grad():
        for sentence,labels in test:
            sent,tags = dataset.gen_tensor(sentence,labels)
            logit = net(sent.view(1,-1))
            logit = logit.squeeze(0)
            loss = loss + criterion(logit, tags).item()
            ptags = torch.argmax(logit, dim=1)
            taglist.extend(  [dataset.i2t[idx.item()] for idx in ptags] )
            correct = (tags==ptags).sum().item()
            hit = hit + correct
            a = a + len(sentence)
        s = ' '.join(taglist)
        s = s + '\n'
        f.write(s)
    print('\nTest: {}/{}'.format(hit,a))
    return hit/a ,loss/dataset.testa_length

# zero epoch
acc,loss = eval(net)
acc_list.append(acc)
loss_list.append(loss)


epoch=2
for i in range(epoch):
    scheduler.step()
    net.train()
    j=0
    for bx , by in dataloader: 
        j = j+1
        sys.stdout.write("Backward progress:%d/%d   \r"%(j,dataset.train_length))
        logit = net( bx ) 
        loss = cross_entropy_for_rnn(logit,by )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    acc,loss = eval(net)
    acc_list.append( acc )
    loss_list.append( loss )
    print("\nIteration:{} loss:{} acc:{}".format(i,loss,acc))

f.close()
f1 = open('acc_and_loss.csv','w')
f1.write('ACC,LOSS\n')
for a,l in zip(acc_list,loss_list):
    s = str(a)+','+str(l)+'\n'
    f1.write(s)
f1.close()
print("保存模型")
torch.save(net.state_dict(), "MODEL/lstm.pth")
