import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ber = [0, 2, 4, 12, 11, 3]
dre = [2, 0, 6, 10, 7, 5]
ham = [4, 6, 0, 8, 15, 1]
kob = [12, 10, 8, 0, 9, 13]
mue = [11, 7, 15, 9, 0, 14]
ros = [3, 5, 1, 13, 14, 0]

dist = pd.DataFrame(data = {"ber": ber, "dre": dre, "ham": ham, "kob": kob, 
                            "mue": mue, "ros": ros}).T

a = (dist**2) * (-0.5)
i = np.diag([1]*6)
u = np.array([1.0]*6)
u = np.reshape(u, (6, 1))
h = i - (1/6 * (u @ u.T))
b = h @ a @ h
e = np.linalg.eigh(b)

g = np.diag(e[0][-2:])
x = e[1][:, -2:] @ (g**0.5)

# Determine the dissimilarities
d12 = ((x[0, 0] - x[1, 0])**2 + (x[0, 1] - x[1, 1])**2)**0.5
d13 = ((x[0, 0] - x[2, 0])**2 + (x[0, 1] - x[2, 1])**2)**0.5
d14 = ((x[0, 0] - x[3, 0])**2 + (x[0, 1] - x[3, 1])**2)**0.5
d15 = ((x[0, 0] - x[4, 0])**2 + (x[0, 1] - x[4, 1])**2)**0.5
d16 = ((x[0, 0] - x[5, 0])**2 + (x[0, 1] - x[5, 1])**2)**0.5

d23 = ((x[1, 0] - x[2, 0])**2 + (x[1, 1] - x[2, 1])**2)**0.5
d24 = ((x[1, 0] - x[3, 0])**2 + (x[1, 1] - x[3, 1])**2)**0.5
d25 = ((x[1, 0] - x[4, 0])**2 + (x[1, 1] - x[4, 1])**2)**0.5
d26 = ((x[1, 0] - x[5, 0])**2 + (x[1, 1] - x[5, 1])**2)**0.5

d34 = ((x[2, 0] - x[3, 0])**2 + (x[2, 1] - x[3, 1])**2)**0.5
d35 = ((x[2, 0] - x[4, 0])**2 + (x[2, 1] - x[4, 1])**2)**0.5
d36 = ((x[2, 0] - x[5, 0])**2 + (x[2, 1] - x[5, 1])**2)**0.5

d45 = ((x[3, 0] - x[4, 0])**2 + (x[3, 1] - x[4, 1])**2)**0.5
d46 = ((x[3, 0] - x[5, 0])**2 + (x[3, 1] - x[5, 1])**2)**0.5

d56 = ((x[4, 0] - x[5, 0])**2 + (x[4, 1] - x[5, 1])**2)**0.5

dd = np.array([[0, d12, d13, d14, d15, d16], [d12, 0, d23, d24, d25, d26], 
               [d13, d23, 0, d34, d35, d36], [d14, d24, d34, 0, d45, d46], 
               [d15, d25, d35, d45, 0, d56], [d16, d26, d36, d46, d56, 0]])

f = [d12, d13, d14, d15, d16, d23, d24, d25, d26, d34, d35, d36, d45, d46, d56]


fig, ax = plt.subplots(figsize = (10, 7))
ax.scatter(range(1, 16), f, c = "w", edgecolor = "k")
ax.plot(range(1, 16), f, linestyle = "--", c = "b")
ax.plot((1, 15), (d12, d56), c = "b")

plt.xlabel("Rank")
plt.ylabel("Distance")
plt.title("Monotonic Regression")
plt.show()