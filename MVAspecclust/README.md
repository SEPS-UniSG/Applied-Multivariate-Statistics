# MVAspecclust
MVAspecclust computes the clusters for the exemplary data based on the Euclidean distance and predefined number of clusters.

```python
# works on pandas 1.5.2, scikit-learn 1.2.0 and matplotlib 3.6.2
import pandas as pd
from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt

data = pd.read_csv("data_example.dat", sep = "\s+", header=None)
data = data.T

sc = SpectralClustering(n_clusters=4, eigen_solver="arpack", affinity = "rbf").fit(data)

fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(data[0], data[1], c = "black")
ax.set_title("Raw Data")
plt.show()

fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(data[0], data[1], c = sc.labels_)
ax.set_title("Derived Clusters")
plt.show()
```
![MVAspecclust](MVAspecclust_1_python.png)
![MVAspecclust](MVAspecclust_2_python.png)