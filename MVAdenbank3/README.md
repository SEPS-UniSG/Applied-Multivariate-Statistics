# MVAdenbank3
Gives plots of the product of univariate and joint kernel density estimates of variables X4 and X5 of the Swiss bank notes.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from KDEpy import FFTKDE
from scipy import stats

xx = pd.read_csv("bank2.dat", sep = "\s+", header=None)

dj_xy, dj_f = FFTKDE(bw = 1.06 * np.array([xx[3].std(), xx[4].std()]) * 200**(-1/5), 
                    kernel='gaussian').fit(np.array(xx.iloc[:,3:5])).evaluate((51, 51))
d1_x, d1_y = FFTKDE(bw=0.3, kernel='gaussian').fit(np.array(xx[3])).evaluate(51)
d2_x, d2_y = FFTKDE(bw=0.3, kernel='gaussian').fit(np.array(xx[4])).evaluate(51)
dp = np.dot(d1_y.reshape(len(d1_y),1), d2_y.reshape(1, len(d2_y)))

X, Y = np.meshgrid(d1_x, d2_x)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, dp.T)
#ax.scatter(x456.iloc[100:, 0], x456.iloc[100:, 1], x456.iloc[100:, 2], c = "w", edgecolors = "r", marker = "^", s = 25)
#ax.set_xlim(6, 14)
#ax.set_ylim(6, 14)
#ax.set_zlim(137, 142)
ax.view_init(elev=0, azim=0)
plt.show()

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(dj_xy[:,0], dj_xy[:,1], np.reshape(dj_f, (2601,1)))
#ax.scatter(x456.iloc[100:, 0], x456.iloc[100:, 1], x456.iloc[100:, 2], c = "w", edgecolors = "r", marker = "^", s = 25)
#ax.set_xlim(6, 14)
#ax.set_ylim(6, 14)
#ax.set_zlim(137, 142)
#ax.view_init(elev=0, azim=0)
plt.show()

fig, ax = plt.subplots()
ax.plot(d1_x, d2_x)
plt.show()
```