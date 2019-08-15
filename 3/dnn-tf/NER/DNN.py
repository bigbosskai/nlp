from utils import load_dataset
import utils
import tensorflow as tf 
import pandas as pd 
import numpy as np 
from tensorflow.contrib.layers import xavier_initializer
import pandas as pd 
import time
#author:bk

f = open('result.txt','w')
np.random.seed(12)
class DNN(object):
    def __init__(self,l2_regularization=True,dropout=True):# please set dropout=? l2 =? in utils
        #是否需要这些
        self.l2_regularization = l2_regularization
        self.dropout = dropout
        self.input_placeholder = tf.placeholder(tf.int32, shape=[None, utils.window_size], name='Input')
        self.labels_placeholder = tf.placeholder(tf.float32, shape=[None, utils.label_size], name='Target')
        self.dropout_placeholder = tf.placeholder(tf.float32, name='Dropout')
        self.load_data()
        self.weights = self.get_weights()
        self.logit,self.cross_entropy = self.build_model(self.weights)#no softmax

        #add loss to total loss
        tf.add_to_collection('total_loss', self.cross_entropy )
        self.loss = tf.add_n(tf.get_collection('total_loss'))

        self.predict = tf.nn.softmax(self.logit) # ok
        self.acc     = self.accuracy()
        self.loss_list = []
        self.acc_list  = []


        self.train_op = tf.train.GradientDescentOptimizer(utils.lr).minimize(self.loss)
        self.init = tf.initialize_all_variables()
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.result = pd.read_excel('TestPredict.xlsx')#存放预测结果的地方

    def accuracy(self):
        correct_prediction = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.labels_placeholder, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return acc

    def load_data(self):
        #读入所有的单词
        word_to_num,num_to_word = utils.load_wv(r'../data/ner/vocab.txt')
        self.wn = len(word_to_num)
        print("读入vocab.txt，总共有{}个单词".format(self.wn))
        print("例如UUUNKKK:{},the:{}".format(word_to_num['UUUNKKK'], word_to_num['the']))
        print("例如0:{},1:{}".format(num_to_word[0], num_to_word[1]))
        # input("Enter")# ok


        # tagnames = ['O','LOC','MISC','ORG','PER']
        tagnames =  ['B-LOC','I-LOC','E-LOC','S-LOC','B-ORG','I-ORG','E-ORG','S-ORG','B-PER','I-PER','E-PER','S-PER','B-MISC','I-MISC','E-MISC','S-MISC','O']


        self.num_to_tag = {i:t for i,t in enumerate(tagnames)}
        self.tag_to_num = {v:k for k,v in self.num_to_tag.items()}
        print("将Tag装换为下标:")
        print(self.num_to_tag)
        print(self.tag_to_num)
        # input("Enter")

        # docs = load_dataset(r'../data/ner/train')
        docs = load_dataset(r'../data/ner/eng.train.bioes')


        self.X_train, self.y_train = utils.docs_to_windows('train',docs,word_to_num,self.tag_to_num,wsize=utils.window_size)
        #
        # self.X_train, self.y_train = utils.increase_postive_samples(self.X_train,self.y_train, self.tag_to_num)
        print("shape:")
        print(self.X_train.shape)
        print(self.y_train.shape)
        # input("Enter")#可怕的数据




        docs = load_dataset(r'../data/ner/eng.testa.bioes')
        self.X_dev, self.y_dev = utils.docs_to_windows('dev',docs,word_to_num,self.tag_to_num,wsize=utils.window_size)
        # np.set_printoptions(threshold=np.inf)
        # print(self.y_dev)
        # input("Enter")

        # docs = load_dataset(r'../data/ner/test.masked')
        # self.X_test, self.y_test = utils.docs_to_windows('test',docs,word_to_num,self.tag_to_num,wsize=utils.window_size)

    def get_weights(self):
        with tf.variable_scope('embedding'):
            ### YOUR CODE HERE
            embedding = tf.get_variable('Embedding', [self.wn, utils.embed_size],initializer=xavier_initializer() )  # assignment中的 L   
        with tf.variable_scope('Layer1'):
            W = tf.get_variable('W',[utils.window_size*utils.embed_size,utils.hidden_size],initializer=xavier_initializer())
            b1= tf.get_variable('b1',[utils.hidden_size])
        if self.l2_regularization:
            tf.add_to_collection('total_loss', 0.5 * utils.l2 * tf.nn.l2_loss(W)) 

        with tf.variable_scope("Layer2"):
            W2 = tf.get_variable('W2',[utils.hidden_size, utils.hidden_size2],initializer=xavier_initializer())
            b2 = tf.get_variable('b2',[utils.hidden_size2])
        if self.l2_regularization:
            tf.add_to_collection('total_loss', 0.5 * utils.l2 * tf.nn.l2_loss(W2)) 
        with tf.variable_scope('Layer3'):
            U = tf.get_variable('U',[utils.hidden_size2,utils.label_size],initializer=xavier_initializer())
            b3= tf.get_variable('b3',[utils.label_size])
        if self.l2_regularization:
            tf.add_to_collection('total_loss', 0.5* utils.l2 * tf.nn.l2_loss(U))
        return {'W':W,'b1':b1,'U':U,'b2':b2,'embedding':embedding,'W2':W2,'b3':b3}

    def build_model(self,weights):
        window = tf.nn.embedding_lookup(weights['embedding'], self.input_placeholder)                
        window = tf.reshape(window, [-1, utils.window_size * utils.embed_size])#(?, 3*50)

        hi = tf.nn.relu(tf.matmul(window, weights['W'])+weights['b1'])
        ho = tf.nn.dropout(hi, self.dropout_placeholder,name='dropout1' ) 

        hi2 = tf.nn.relu( tf.matmul(hi, weights['W2']) + weights['b2'])
        ho2 = tf.nn.dropout(hi2, self.dropout_placeholder, name='dropout2')

        logit = tf.matmul(ho2,weights['U']) + weights['b3']
        cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=self.labels_placeholder))
        return logit,cross_entropy
    def next_batch(self,x_train,y_train,batch_size):
        n_samples = x_train.shape[0]
        ri = np.random.permutation( n_samples )
        x_train = x_train[ri] # shuffle data
        y_train = y_train[ri]
        for i in range(0,n_samples,batch_size):
            upper_bound = min(i+batch_size,n_samples)
            ret_x = x_train[i:upper_bound]
            ret_y = y_train[i:upper_bound]
            yield ret_x,ret_y
    def fit(self,epochs=100,batch_size=100):
        loss,acc,predict = self.sess.run([self.loss,self.acc,self.predict],
                                         feed_dict=
                                         {
                                            self.input_placeholder:self.X_dev,
                                            self.labels_placeholder:self.y_dev,
                                            self.dropout_placeholder:1.0
                                            })
        pdx = np.argmax(predict, axis=1)
        predTag = []
        for i in pdx:
            predTag.append( self.num_to_tag[i])
        # self.result['Iter'+'0'] = predTag
        s = ' '.join(predTag) +'\n'
        f.write(s)
        # self.result.to_csv('dev.csv',sep=',',index=False)
        print("Iteration:{} loss:{} acc:{}".format(0,loss,acc))
        self.loss_list.append( loss )
        self.acc_list.append( acc )
        for i in range(1,epochs+1):
            for bx,by in self.next_batch(self.X_train,self.y_train,batch_size):
                _ = self.sess.run([self.train_op], 
                                  feed_dict=
                                  {
                                    self.input_placeholder:bx, 
                                    self.labels_placeholder:by,
                                    self.dropout_placeholder: utils.dropout
                                    })
            loss,acc,predict = self.sess.run([self.loss,self.acc,self.predict],
                                            feed_dict=
                                            {
                                                self.input_placeholder:self.X_dev, 
                                                self.labels_placeholder:self.y_dev,
                                                self.dropout_placeholder:1.0
                                                })
            pdx = np.argmax(predict, axis=1)
            predTag = []
            for j in pdx:
                predTag.append( self.num_to_tag[j])
            s = ' '.join(predTag) +'\n'
            f.write(s)
            # self.result['Iter'+str(i)] = predTag
            # self.result.to_csv('dev.csv',sep=',',index=False)
            print("Iteration:{} loss:{} acc:{}".format(i,loss,acc))
            self.loss_list.append( loss )
            self.acc_list.append( acc )

    def dump(self,dump_file):
        df = pd.DataFrame({'LOSS':self.loss_list,'ACC':self.acc_list})
        df.to_excel(dump_file,index=False)
        st = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) 
        self.result.to_excel(st+'_TestPredict.xlsx', index=False)
        print("Save to file: "+dump_file)



def main():
    dnn =DNN()
    dnn.fit(250)
    dnn.dump('LossAndAcc.xlsx')


if __name__ == '__main__':
    main()
    f.close()
