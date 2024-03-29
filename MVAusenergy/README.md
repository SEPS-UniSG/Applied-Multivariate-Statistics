# MVAusenergy
Computes a test statistic for the variance of the US energy company data. H0:
empirical variance is equal to the theoretical variance, given unknown mean.

```python
# works on numpy 1.24.3 and scipy 1.10.1
import numpy as np
from scipy.stats import chi2

#energy sector data
x = np.array([[13621, 4848, 4572, 485, 898.9, 23.4], [1117, 1038, 478, 59.7, 91.7, 3.8],
[1633, 701, 679, 74.3, 135.9, 2.8], [5651, 1254, 2002, 310.7, 407.9, 6.2],
[5835, 4053, 1601, -93.8, 173.8, 10.8], [3494, 1653, 1442, 160.9, 320.3, 6.4],
[1654, 451, 779, 84.8, 130.4, 1.6], [1679, 1354, 687, 93.8, 154.6, 4.6],
[1257, 355, 181, 167.5, 304, 0.6], [1743, 597, 717, 121.6, 172.4, 3.5],
[1440, 1617, 639, 81.7, 126.4, 3.5], [14045, 15636, 2754, 418, 1462, 27.3],
[3010, 749, 1120, 146.3, 209.2, 3.4], [3086, 1739, 1507, 202.7, 335.2, 4.9],
[1995, 2662, 341, 34.7, 100.7, 2.3]])

S = np.cov(x[:, :2], rowvar = False) # unbiased variance matrix for energy sector
s = S * 14 / 15 

#manufacturing sector data
y = np.array([[1093, 1679, 1070, 100.9, 164.5, 20.8], [1128, 1516, 430, -47, 26.7, 13.2],
[1804, 2564, 483, 70.5, 164.9, 26.6], [4662, 4781, 2988, 28.7, 371.5, 66.2],
[6307, 8199, 598, -771.5, -524.3, 57.5], [2366, 3305, 1117, 131.2, 256.5, 25.2],
[4084, 4346, 3023, 302.7, 521.7, 37.5], [10348, 5721, 1915, 223.6, 322.5, 49.5],
[752, 2149, 101, 11.1, 15.2, 2.6], [10528, 14992, 5377, 312.7, 710.7, 184.8]])

T = np.cov(y[:, :2], rowvar = False) # unbiased variance matrix for manufacturing sector
sigma = T * (9 /10)

test = 15 * np.sum(np.diag(np.linalg.solve(sigma, s))) - 15 * np.log10(np.linalg.det(np.linalg.solve(sigma, s))) - 15 * 2
p = 1 - chi2.cdf(test, 3)

print("Test Statistic = ", np.round(test, 3))
print("p-value = ", np.round(p, 3))
```
