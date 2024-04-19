# MVAtdistail
Plots four tails of probability density functions of the t-distribution and one tail
of the Gauss distribution with different degrees of freedom (t3 stands for t-distribution with 3
degrees of freedom, etc).

```python
# works on numpy 1.23.5, matplotlib 3.6.2 and scipy 1.10.0
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t, norm

xx = np.arange(2.5, 4.1, 0.1)

fig, ax = plt.subplots(1,1,figsize=(10,10))

ax.plot(xx, norm.pdf(xx), 'grey', lw=3, label='Gaussian')
ax.plot(xx, t.pdf(xx, 1), 'black', lw=3, label='t1')
ax.plot(xx, t.pdf(xx, 3), 'blue', lw=3, label='t3')
ax.plot(xx, t.pdf(xx, 9), 'red', lw=3, label='t9')
ax.plot(xx, t.pdf(xx, 45), 'purple', lw=3, label='t45')

ax.set_xlim([2.5, 4])
ax.set_ylim([0, 0.04])
ax.set_xlabel('X', fontsize=20)
ax.set_ylabel('Y', fontsize=20)
ax.set_title('Tail comparison of t-distribution', fontsize=20)
ax.legend(loc='upper right', fontsize=16)

plt.show()
```
![MVAtdistail](MVAtdistail_python.png)

