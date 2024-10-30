# MVAregbank
Computes a linear regression of column 5 (upper inner frame) and column 4 (lower
inner frame) for the genuine Swiss bank notes.

```python
# works on pandas 1.5.2, numpy 1.23.5 and matplotlib 3.6.3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

x = pd.read_csv("bank2.dat", sep = "\s+", header=None)

x1 = x.iloc[:100,3]
y = x.iloc[:100,4]

a, b = np.polyfit(x1,y,1)
xfit = np.linspace(x1.min(),x1.max(),400)
yfit = b + a*xfit

fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(x1, y, c = "black")
ax.plot(xfit,yfit, c = "r")
plt.xlabel("Lower inner frame(X4), genuine", fontsize = 20)
plt.ylabel("Upper inner frame(X5), genuine", fontsize = 20)
plt.title("Swiss bank notes", fontsize = 25, weight="bold")

plt.show()
```
![MVAregbank](MVAregbank_python.png)