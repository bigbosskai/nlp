import pandas as pd 
df = pd.read_excel('TestPredict.xlsx')
#word Real LSTMPredict
entities = ['B-LOC','I-LOC','E-LOC','S-LOC','B-ORG','I-ORG','E-ORG','S-ORG','B-PER','I-PER','E-PER','S-PER','B-MISC','I-MISC','E-MISC','S-MISC']


en = df[df['Real'].isin(entities)]
TP = en[ en['Real']==en['DNNPredict'] ].shape[0]

fptmp = df[ df['Real']=='O']
FP = fptmp[ fptmp['DNNPredict'].isin(entities)].shape[0]
FN = en[ en['DNNPredict']=='O'].shape[0]

print(TP)
print(FP)
print(FN)

precision = TP/(TP+FP)
recall    = TP/(TP+FN)
print(precision,recall)