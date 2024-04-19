#works on statsmodels 0.14 and matplotlib 3.8.0
import matplotlib.pyplot as plt
from statsmodels.distributions.copula.api import ClaytonCopula, GumbelCopula

sample_size = 10000
clay = ClaytonCopula(theta=3, k_dim=2).rvs(sample_size, random_state= 1)
gum = GumbelCopula(theta=3, k_dim=2).rvs(sample_size, random_state= 1)

plt.figure(figsize=(8, 8))
plt.scatter(clay[:, 0], clay[:, 1], c='blue', s=2)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Clayton Copula")
plt.show()

plt.figure(figsize=(8, 8))
plt.scatter(gum[:, 0], gum[:, 1], c='blue', s=2)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Gumbel Copula")
plt.show()
