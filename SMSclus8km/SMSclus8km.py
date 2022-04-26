import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans

random.seed(100)

eight = np.array(([-3, -2, -2, -2, 1, 1, 2, 4], [0, 4, -1, -2, 4, 2, -4, -3])).T
eight = eight[[7,6,2,0,3,1,5,4], :]
results = KMeans(n_clusters=2, random_state=100, algorithm = "full").fit(eight)

fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(results.cluster_centers_[0, 0], results.cluster_centers_[0, 1], 
           c = "w", edgecolor = "black", zorder = 10)
ax.scatter(results.cluster_centers_[1, 0], results.cluster_centers_[1, 1], 
           c = "w", edgecolor = "black", zorder = 10)
for i in range(0, 8):
    ax.plot([eight[i, 0], results.cluster_centers_[results.labels_[i], 0]], 
            [eight[i, 1], results.cluster_centers_[results.labels_[i], 1]], 
            c = "black")
ax.plot([results.cluster_centers_[0, 0], results.cluster_centers_[1, 0]], 
        [results.cluster_centers_[0, 1], results.cluster_centers_[1, 1]], 
        c = "black")

for i in range(0, 8):
    ax.scatter(eight[i, 0], eight[i, 1], c = "w", edgecolor = "black", s = 300, 
               zorder = 10)
    ax.text(eight[i, 0], eight[i, 1], str(i+1), zorder = 15, 
            horizontalalignment='center', verticalalignment='center', c = "r", 
            fontsize = 12)
ax.set_xlabel("price conciousness")
ax.set_ylabel("brand loyalty")
plt.title("8 points - k-means clustering")
    
plt.show()