import pandas as pd 

with open('result.txt','r') as f:
	lines = f.readlines()
	tks = lines[-1].strip().split(' ')

df = pd.read_excel('TestPredict.xlsx')

df['DNNPredict'] = tks

df.to_excel('TestPredict2.xlsx', index=False)