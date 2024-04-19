# MVAnmdscar1
Plots the initial configuration of the points for nonmetric MDS for car marks data.

```python
# works on numpy 1.23.5 and matplotlib 3.6.2
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[3, 2, 1, 10], [2, 7, 3, 4]]).T
brands = ["Mercedes", "Jaguar", "Ferrari", "VW"]

fig, ax = plt.subplots(figsize = (7, 7))
ax.scatter(x[:, 0], x[:, 1], c = "w")
for i in range(0, len(x)):
    ax.text(x[i, 0]+0.05, x[i, 1]+0.1, brands[i], fontsize = 14)

plt.xlim(0, 12)
plt.ylim(0, 8)

plt.title("Initial Configuration", fontsize = 16)
plt.show()
```
![MVAnmdscar1](MVAnmdscar1_python.png)