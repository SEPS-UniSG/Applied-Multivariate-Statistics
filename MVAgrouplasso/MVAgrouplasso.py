import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from group_lasso import LogisticGroupLasso

# Load the data
splice = pd.read_csv('splice.csv')
X = splice.iloc[:,1:8]
y = splice.iloc[:,0]

X_category = pd.get_dummies(X, dtype=int)
#lambdas = np.linspace(80,0,100)
groups = np.repeat([1,2,3,4],7)
#lmb = 0.05



#for lmb in lambdas:
    #gl = LogisticGroupLasso(groups= groups, l1_reg= lmb).fit(X_category,y)
#glc = linear_model.LogisticRegression().fit(X,y)


