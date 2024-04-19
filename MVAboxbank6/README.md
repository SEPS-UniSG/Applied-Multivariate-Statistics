# MVAboxbank6
Computes boxplots for the diagonal (X6 variable) of the genuine and forged banknotes from the Swiss bank data.

```python
# works on pandas 1.5.2 and matplotlib 3.6.3
import pandas as pd
import matplotlib.pyplot as plt

x = pd.read_csv("bank2.dat", sep = "\s+", header=None)


fig, ax = plt.subplots(figsize = (10, 10))
ax.boxplot([x.iloc[:100, 5], x.iloc[100:200, 5]], labels = ["GENUINE", "COUNTERFEIT"], 
           meanline = True, showmeans = True)
plt.title("Swiss Bank Notes")

plt.show()
```
![MVAboxbank6](MVAboxbank6_python.png)