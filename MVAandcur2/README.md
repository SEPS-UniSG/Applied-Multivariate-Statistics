# MVAandcur2
Computes Andrew''s Curves for the observations 96-105 of the Swiss bank notes data. The order of the variables is 6,5,4,3,2,1.

```python
# works on pandas 1.5.2, numpy 1.24.1 and matplotlib 3.6.3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("bank2.dat", sep = "\s+", header=None)
x = data[95:105]

y = (x - x.min())/(x.max() - x.min())


def ac(x, t):
    if len(x) % 2 == 0:
        f = x[0]/np.sqrt(2)
        u = 1
        for p in range(1, int((len(x)+1)/2)):
            f += x[u]*np.sin(t*p) + x[u+1]*np.cos(t*p)
            u += 2
        f += x[u]*np.sin(t*(p+1))
        return f
    else:
        f = x[0]/np.sqrt(2)
        u = 1
        for p in range(1, int((len(x)+1)/2)):
            f += x[u]*np.sin(t*p) + x[u+1]*np.cos(t*p)
            u += 2
        return f


grid = np.linspace(0, 2*np.pi, 1000)

fig, ax = plt.subplots(figsize = (15,10))
for i in range(0, 5):
    ax.plot(grid, ac(list(reversed(y.iloc[i,:])), grid), c = "black")
    
for i in range(5, 10):
    ax.plot(grid, ac(list(reversed(y.iloc[i,:])), grid), c = "red", ls = "--")

plt.title("Andrews curves (Bank data)")
ax.set_xticklabels(list(reversed(list(range(0, 8)))))

plt.show()


```
![MVAandcur](MVAandcur2_python.png)