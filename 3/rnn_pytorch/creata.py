import pandas as pd 

df = pd.read_excel('TestPredict.xlsx')
with open('predict.txt','r') as f:
	lines = f.readlines()
	Tags = lines[-1].strip().split(' ')

df['Predict'] = Tags

df.to_excel('TestPredict2.xlsx', index=False)
print(df)
print( len(Tags))