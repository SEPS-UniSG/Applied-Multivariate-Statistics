# MVAscabank45
Computes a two dimensional scatterplot of X4 vs. X5 (upper inner frame vs. lower) of the Swiss bank notes data.

```python
# works on pandas 1.5.2 and matplotlib 3.6.2
import pandas as pd
import matplotlib.pyplot as plt

x = pd.read_csv("bank2.dat", sep = "\s+", header=None)

fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(x.iloc[:,3], x.iloc[:,4], c = "w", edgecolors = "black")
plt.xlim(7, 13)
plt.ylim(7, 13)
plt.title("Swiss bank notes")
plt.show()
```
![MVAscabank45](MVAscabank45_python.png)