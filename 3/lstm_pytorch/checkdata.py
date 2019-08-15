import utils
import numpy as np 
np.set_printoptions(threshold=np.inf)
data = utils.Data('eng.train.bioes','eng.testa.bioes','vocab.txt')
def to_word(idxvec, n2w):
	return [n2w[idx] for idx in idxvec]
#['B-LOC','I-LOC','E-LOC','S-LOC','B-ORG','I-ORG','E-ORG','S-ORG','B-PER','I-PER','E-PER','S-PER','B-MISC','I-MISC','E-MISC','S-MISC','O']
print(data.X_test[0])
print(data.test_sequence_length[0])
print(to_word(data.X_test[0] , data.n2w))
print(data.Y_test[0])