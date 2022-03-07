# MVAcanus
Performs a canonical correlation analysis for the US crime and US health data.

```python
import pandas as pd
import numpy as np

X = pd.read_csv("uscrime.dat", sep = "\s+", header=None)
X = X.iloc[:, [2, 3, 4, 5, 6, 7, 8]]

Y1 = pd.read_csv("ushealth.dat", sep = "\s+", header=None)
Y2 = Y1.iloc[:, [3, 4, 5, 6, 7, 8, 9]]

# Estimation of covariance matrices
S = np.cov(np.hstack((np.array(X), np.array(Y2))).T)
Sxx = S[:len(X.columns), :len(X.columns)]
Sxy = S[:len(X.columns), len(Y2.columns):(len(X.columns) + len(Y2.columns))]
Syx = Sxy
Syy = S[len(Y2.columns):(len(X.columns) + len(Y2.columns)), len(Y2.columns):(len(X.columns) + len(Y2.columns))]

# Estimation of the matrix K and its singular value decomposition
eigenX = np.linalg.eig(Sxx)
eX = eigenX[0]
vX = eigenX[1]

eigenY = np.linalg.eig(Syy)
eY = eigenY[0]
vY = eigenY[1]

K = vX @ np.diag(1/np.sqrt(eX)) @ vX.T @ Sxy @ vY @ np.diag(1/np.sqrt(eY)) @ vY.T

G, L, D = np.linalg.svd(K, full_matrices = False)
L = np.diag(L)

# Estimated canonical correlation vectors (a and b) and canonical variables (eta and phi)
a = vX @ np.diag(1/np.sqrt(eX)) @ vX.T @ G
b = vY @ np.diag(1/np.sqrt(eY)) @ vY.T @ D.T

eta = X @ a
phi = Y2 @ b
```