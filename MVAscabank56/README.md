# MVAscabank56
Computes a two dimensional scatterplot of X5 vs. X6 (upper inner frame vs. diagonal) of the Swiss bank notes.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = pd.read_csv("bank2.dat", sep = "\s+", header=None)
x56 = x.iloc[:,4:]
x1 = [1] * 100
x2  = [2] * 100
xx = x56.copy()
x1.extend(x2)
xx["x1x2"] = x1

fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(xx.iloc[:100,0], xx.iloc[:100,1], c = "w", edgecolors = "black")
ax.scatter(xx.iloc[100:,0], xx.iloc[100:,1], c = "w", edgecolors = "r", marker = "^")
plt.xlim(7, 13)
plt.ylim(137.5, 142.5)
plt.yticks(list(np.arange(137.5, 143, 1)))
plt.title("Swiss bank notes")
plt.show()
```
![MVAscabank56](MVAscabank56_python.png)