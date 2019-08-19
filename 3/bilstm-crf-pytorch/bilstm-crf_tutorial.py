import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import numpy as np 
import sys
from utils import CONLL2003
import utils

torch.manual_seed(1997)#设置随机数种子
np.random.seed(1997)

class Config(object):
    word_embed_size = 10
    batch_size = 5 # 每批处理句子的数目
    label_size = 17  # 类别数 y 的纬度
    lstm_hidden_size = 20   # 
    lr          = 0.2

def argmax(vec):
    #返回python int
    _, idx = torch.max(vec, 1)#对1维度求最大值
    return idx.item()

def prepare_sequence(seq, to_ix):#seq = [w0,w1,w2,w3 ...]  
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]#tensor(value)返回的是一个值
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix):#这里的hidden_dim设置为偶数
        super(BiLSTM_CRF, self).__init__()
        self.config = Config()

        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.embedding_dim = self.config.word_embed_size
        self.hidden_dim = self.config.lstm_hidden_size
        
        self.tagset_size = len(tag_to_ix)

        #设置embedding的矩阵
        self.word_embeds = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True)
        
        # 把LSTM的输出映射到label空间
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

        # 设置转移矩阵 shape是T*T,T已经包含START_TAG 和STOP_TAG在里面了
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))#来自N(0,1)

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]]  = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))

    def _forward_alg(self, feats):
        # Do the forward algorithm to compute the partition function
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        # START_TAG has all of the score.
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas

        # Iterate through the sentence
        for feat in feats:
            alphas_t = []  # The forward tensors at this timestep
            for next_tag in range(self.tagset_size):
                # broadcast the emission score: it is the same regardless of
                # the previous tag
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                # the ith entry of trans_score is the score of transitioning to
                # next_tag from i
                trans_score = self.transitions[next_tag].view(1, -1)
                # The ith entry of next_tag_var is the value for the
                # edge (i -> next_tag) before we do log-sum-exp
                next_tag_var = forward_var + trans_score + emit_score
                # The forward variable for this tag is log-sum-exp of all the
                # scores.
                alphas_t.append(log_sum_exp(next_tag_var).view(1))
            forward_var = torch.cat(alphas_t).view(1, -1)
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)#embeds:[seq, 1, embedding_dim]
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)
        # feats: [seq, tagset_size]
        # 下面是feats每一行的解读
        # w0: [0.1, 0.23, 1.5, ..., 2.0] 有tagset_size个标签
        # w1: [0.1, 0.23, 1.5, ..., 2.0] 有tagset_size个标签
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score#最小化这个

    def forward(self, sentence):
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        # Find the best path, given the features.
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq




START_TAG = "<START>"
STOP_TAG = "<STOP>"


dataset    = CONLL2003('./data')
dataloader = DataLoader( dataset, batch_size=1, shuffle=False,num_workers=0, drop_last=False)

word_to_ix = dataset.w2i

tag_to_ix = dataset.t2i

vocab_size = len(dataset.w2i)

model = BiLSTM_CRF(  vocab_size, tag_to_ix)
# optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)
optimizer = optim.SGD(model.parameters(),lr= 0.2)
scheduler = lr_scheduler.StepLR(optimizer,step_size=5,gamma = 0.96)


total_sum,trainable_sum = utils.get_parameter_number(model)
print("参数总数,训练参数")
print(total_sum,trainable_sum)

f=open('predict.txt','w')
acc_list = []

def eval(DATASET):
    test = DATASET.testa
    tags_list = []
    ALL = 0.
    HIT = 0.
    with torch.no_grad():
        for sentence,labels in test:
            sent,tags = DATASET.gen_tensor(sentence,labels)
            ptags = model(sent)[1]#返回的是list
            # 我要将其转化为string
            Stags = [dataset.i2t[p] for p in ptags]
            ptags = torch.tensor(ptags, dtype=torch.long)
            correct = (tags==ptags).sum().item()
            HIT = HIT + correct
            ALL = ALL + len(sentence)
            tags_list.extend(Stags)
    s = ' '.join(tags_list)
    s = s + '\n'
    f.write(s)

    print('\nTest: {}/{}'.format(HIT,ALL))
    return HIT/ALL
acc = eval(dataset)
acc_list.append(acc)
epochs = 20
# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(epochs):
    scheduler.step()
    #在这里统计ACC
    print('\nEpoch:{} ACC:{}'.format(epoch,acc) )
    for i, data in enumerate(dataloader):
        sys.stdout.write("Backward progress: %d/%d   \r" % (i,dataset.train_length))
        sentence,labels = data
        sentence = sentence.squeeze(0)
        labels   = labels.squeeze(0)
        model.zero_grad()
        loss = model.neg_log_likelihood(sentence, labels)
        loss.backward()
        optimizer.step()
    acc=eval(dataset)
    acc_list.append(acc)
    print('\nEpoch:{} ACC:{}'.format(epoch,acc) )
f.close()
f1 = open('acc_and_loss.csv','w')
f1.write('ACC\n')
for a in acc_list:
    s = str(a)+'\n'
    f1.write(s)
f1.close()
print("保存模型")
torch.save(net.state_dict(), "MODEL/bilstm-crf.pth")
