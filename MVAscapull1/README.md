# MVAscapull1
Computes a two dimensional scatterplot of price and sales from the pullovers data set.

```python
# works on pandas 1.5.2 and matplotlib 3.6.2
import pandas as pd
import matplotlib.pyplot as plt

x = pd.read_csv("pullover.dat", sep = "\s+", header=None)

fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(x.iloc[:,1], x.iloc[:,0], c = "w", edgecolors = "black")
plt.xlim(78, 127)
plt.ylim(80, 240)
plt.xlabel("Price (X2)", fontsize = 14)
plt.ylabel("Sales (X1)", fontsize = 14)
plt.title("Pullovers Data", fontsize = 14)
plt.show()
```
![MVAscapull1](MVAscapull1_python.png)