import pandas as pd
import numpy  as np
import matplotlib.pyplot as plt

df = pd.read_excel('LossAndAcc.xlsx')
steps = df.shape[0]
epochs = [i for i in range(steps)]

# lfm_v = [lfm_n for i in range(100) ]
fig1 = plt.figure( figsize=(4.8,4.5),dpi=80)

plt.plot(epochs, df['LOSS'], 'r-',label='Loss(Test)',linewidth=1.3)
plt.plot(epochs, df['ACC'] , 'b-',label='Accuracy(Test)',linewidth=1.3)

# plt.xlim(4,26)
plt.ylim(0.4,1.1)
plt.legend(loc='upper right')
# plt.yticks(np.around(np.linspace(0.9,1.1,6) ,decimals=2))
#plt.xticks(epochs)
plt.xlabel('Epoch')
# plt.ylabel('RMSE')
# plt.title('MovieLens')
plt.grid(True,axis='y')
plt.show()

