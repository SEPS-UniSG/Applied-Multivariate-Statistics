# MVAaerbh

```
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy

data = pd.read_csv("bostonh.dat", sep = "\s+", header=None)

# transform data

xt = data
for i in [0, 2, 4, 5, 7, 8, 9, 13]:
    xt.iloc[:, i] = np.log(data.iloc[:, i])
    
xt.iloc[:, 1] = data.iloc[:, 1]/10
xt.iloc[:, 6] = (data.iloc[:, 6]**(2.5))/10000
xt.iloc[:, 10] = np.exp(0.4 * data.iloc[:, 10])/1000
xt.iloc[:, 11] = data.iloc[:, 11]/100
xt.iloc[:, 12] = np.sqrt(data.iloc[:, 12])
data = xt.drop(3, axis = 1)

da = (data - np.mean(data))/np.std(data, ddof = 1)
d = np.zeros([len(da),len(da)])

for i in range(0, len(da)):
    for j in range(0, len(da)):
        d[i, j] = np.linalg.norm(da.iloc[i, :] - da.iloc[j, :])       
        
ddd  = d[1:, :-1][:, 0]
for i in range(1, len(da)-1):
    ddd = np.concatenate((ddd, d[1:, :-1][i:, i]))

w = hierarchy.linkage(ddd, 'ward')
tree = hierarchy.cut_tree(w, n_clusters = 2)

n = len(data)

# AER for clusters of Boston houses

da["tree"] = tree

mis1  = 0
mis2  = 0
corr1 = 0
corr2 = 0

for i in range(n):
    dai = da.drop(i)
    t1 = dai[dai["tree"] == 0].iloc[:, :-1]
    t2 = dai[dai["tree"] == 1].iloc[:, :-1]
    m1 = t1.mean()
    m2 = t2.mean()
    m = (m1 + m2)/2
    s = ((len(t1) - 1) * np.cov(t1.T) + (len(t2) - 1) * np.cov(t2.T))/(len(da) - 2)
    alpha = np.linalg.inv(s) @ (m1 - m2)
    mis1 = mis1 + int(tree[i] == 0) * int((da.iloc[i, :-1] - m) @ alpha < 0)
    mis2 = mis2 + int(tree[i] == 1) * int((da.iloc[i, :-1] - m) @ alpha > 0)
    corr1 = corr1 + int(tree[i] == 0) * int((da.iloc[i, :-1] - m) @ alpha > 0)
    corr2 = corr2 + int(tree[i] == 1) * int((da.iloc[i, :-1] - m) @ alpha < 0)
    
aer = (mis1 + mis2)/len(da)            # AER (actual error rate)
print(aer) 

```
