import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from group_lasso import LogisticGroupLasso


splice = pd.read_csv('splice.csv')
X = splice.iloc[:,1:8]
y = splice.iloc[:,0]
X_category = pd.get_dummies(X, dtype=int)

lambdas = np.linspace(0,0.05,20)
groups = np.repeat([1,2,3,4,5,6,7],4)
path = np.zeros((len(lambdas), X_category.shape[1]))

fig,ax = plt.subplots(figsize=(8,6))

for i, lmb in enumerate(lambdas):
    gl = LogisticGroupLasso(groups= groups, 
                            l1_reg= lmb, 
                            tol=1e-3,
                            scale_reg=None, 
                            supress_warning=True).fit(X_category,y)
    coefs = gl.coef_
    path[i, :] = coefs.T[0]

for i in range(path.shape[1]):
    ax.plot(lambdas,path[:,i])

ax.set_xlabel('Lambda')
ax.set_ylabel('Coefficients')
ax.invert_xaxis()
ax.set_title('Coefficient paths')
plt.show()





