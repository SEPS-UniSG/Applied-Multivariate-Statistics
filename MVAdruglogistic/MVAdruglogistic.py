import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.stats import chi2

# Drug data
zi = np.array([
    [1, 0, 1, 0, 1, 0, 0, 0, 0, 21],
    [1, 0, 1, 0, 0, 1, 0, 0, 0, 32],
    [1, 0, 1, 0, 0, 0, 1, 0, 0, 70],
    [1, 0, 1, 0, 0, 0, 0, 1, 0, 43],
    [1, 0, 1, 0, 0, 0, 0, 0, 1, 19],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 683],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 596],
    [1, 0, 0, 1, 0, 0, 1, 0, 0, 705],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 295],
    [1, 0, 0, 1, 0, 0, 0, 0, 1, 99],
    [0, 1, 1, 0, 1, 0, 0, 0, 0, 46],
    [0, 1, 1, 0, 0, 1, 0, 0, 0, 89],
    [0, 1, 1, 0, 0, 0, 1, 0, 0, 169],
    [0, 1, 1, 0, 0, 0, 0, 1, 0, 98],
    [0, 1, 1, 0, 0, 0, 0, 0, 1, 51],
    [0, 1, 0, 1, 1, 0, 0, 0, 0, 738],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 700],
    [0, 1, 0, 1, 0, 0, 1, 0, 0, 847],
    [0, 1, 0, 1, 0, 0, 0, 1, 0, 336],
    [0, 1, 0, 1, 0, 0, 0, 0, 1, 196]
])

y = zi[:, 9]

I, J, K = 2, 2, 5
average = np.array([[23.2], [36.5], [54.3], [69.2], [79.5], [23.2], [36.5], [54.3], [69.2], [79.5]])
X = np.array([
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, 1],
    [1, -1],
    [1, -1],
    [1, -1],
    [1, -1],
    [1, -1]
])

X1 = np.hstack((X, average))

n1jk = y[zi[:, 2] == 1]
n2jk = y[zi[:, 2] == 0]

b0 = np.zeros(X1.shape[1])

def ff(b0):
    p1 = np.exp(X1 @ b0) / (1 + np.exp(X1 @ b0))
    p2 = 1 - p1
    return -np.sum(n1jk * np.log(p1) + n2jk * np.log(p2))

b = opt.minimize(ff, b0).x

p1 = np.exp(X1 @ b) / (1 + np.exp(X1 @ b))
p2 = 1 - p1
nfit = np.hstack(((n1jk + n2jk) * p1, (n1jk + n2jk) * p2))
nobs = np.concatenate((n1jk, n2jk))

e = np.log(nobs) - np.log(nfit)

df = X1.shape[0] - X1.shape[1]
G2 = 2 * np.sum(nobs * e)
pvalG2 = 1 - chi2.cdf(G2, df)

chi2 = np.sum(((nobs - nfit) ** 2) / nfit)

print("Degree of freedom:", df)
print("G2:", G2)
print("p-value G2:", pvalG2)
print("Chi-square:", chi2)

# Plotting code
oddratfit = np.log(p1 / p2)
oddrat = np.log(n1jk / n2jk)

plt.plot(X1[:K, -1], oddratfit[:K], label="Fitted - Drug Yes")
plt.plot(X1[K:2 * K, -1], oddratfit[K:2 * K], label="Fitted - Drug No")
plt.plot(X1[:K, -1], oddrat[:K], 'r*', markersize=10, label="Observed - Drug Yes")
plt.plot(X1[K:2 * K, -1], oddrat[K:2 * K], 'b*', markersize=10, label="Observed - Drug No")
plt.xlabel("Age category")
plt.ylabel("Log of odds-ratios")
plt.ylim(-3.5, -0.5)
plt.title("Fit of the log of the odds-ratios")
plt.legend()
plt.show()

