# works on numpy 1.23.5, pandas 1.5.2, scipy 1.9.3 and matplotlib 3.6.2
import pandas as pd
import numpy as np
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

data = pd.read_csv("bostonh.dat", sep = "\s+", header=None)

# transform data
xt = pd.DataFrame({
    0: np.log(data.iloc[:, 0]),
    1: data.iloc[:, 1] / 10,
    2: np.log(data.iloc[:, 2]),
    3: data.iloc[:, 3],
    4: np.log(data.iloc[:, 4]),
    5: np.log(data.iloc[:, 5]),
    6: (data.iloc[:, 6]**2.5) / 10000,
    7: np.log(data.iloc[:, 7]),
    8: np.log(data.iloc[:, 8]),
    9: np.log(data.iloc[:, 9]),
    10: np.exp(0.4 * data.iloc[:, 10]) / 1000,
    11: data.iloc[:, 11] / 100,
    12: np.sqrt(data.iloc[:, 12]),
    13: np.log(data.iloc[:, 13])
})

data = xt.drop(3, axis = 1)

da = (data - np.mean(data))/np.std(data, ddof = 1)
d = np.linalg.norm(da.values[:, np.newaxis] - da.values, axis = 2)

ddd = np.hstack([d[i+1:, i] for i in range(len(da)-1)])

w = hierarchy.linkage(ddd, 'ward')
tree = hierarchy.cut_tree(w, n_clusters = 2)

da["tree"] = tree

t1 = da[da["tree"] == 0].iloc[:, :-1]
t2 = da[da["tree"] == 1].iloc[:, :-1]

# Plot 1
fig, ax = plt.subplots(figsize = (10, 10))
h = hierarchy.dendrogram(w)
plt.title("Ward method", fontsize = 16)
plt.ylabel("distance", fontsize = 16)
ax.set_xticks([])
plt.show()

# means for Cluster 1 and Cluster 2
mc = np.vstack((np.mean(da[da["tree"] == 0]), np.mean(da[da["tree"] == 1])))
# standard deviations for Cluster 1 and Cluster 2
sc = np.vstack((np.std(da[da["tree"] == 0].values, ddof = 1),
                np.std(da[da["tree"] == 1].values, ddof = 1)))
# means and standard deviations of the 13 standardized variables for Cluster 1
# (249 observations) and Cluster 2 (257 observations)
tbl = np.vstack((mc[0,:], np.hstack(([sc[0,:]/np.sqrt(len(t1))] * len(mc.T))), 
                 mc[1,:], np.hstack(([sc[1,:]/np.sqrt(len(t2))] * len(mc.T)))))

# spectral decomposition
eva, eve = np.linalg.eig(da.iloc[:, :-1].cov())
eve = eve[:, :2]

dav = da.iloc[:, :-1] @ eve
dav["tr"] = tree

# Plot 2
fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(dav[dav["tr"] == 0][0], -dav[dav["tr"] == 0][1], c = "red")
ax.scatter(dav[dav["tr"] == 1][0], -dav[dav["tr"] == 1][1], c = "black", marker = "^")
plt.title("First vs. Second PC", fontsize = 16)
plt.xlabel("PC1", fontsize = 16)
plt.ylabel("PC2", fontsize = 16)
plt.show()

#  Plot 3
dat = (xt - np.mean(xt))/np.std(xt, ddof = 1)
dat["tree"] = tree
t3 = dat[dat["tree"] == 0]
t4 = dat[dat["tree"] == 1]

fig, axes = plt.subplots(2, 7, figsize = (15, 10))
v = 1
for i in range(0, 7):
    axes[0, i].boxplot([t3[i], t4[i]], widths=(0.5, 0.5))
    axes[0, i].set_xlabel("X" + str(v))
    v += 1
    
for i in range(0, 7):
    axes[1, i].boxplot([t3[i+7], t4[i+7]], widths=(0.5, 0.5))
    axes[1, i].set_xlabel("X" + str(v))
    v += 1
plt.show()


#  Plot 4
xt["tr"] = tree
fig, axes = plt.subplots(7, 7, figsize = (15,15))

for i in range(0, 7):
    for j in range(0, 7):
        if i == j:
            axes[i, j].text(0.5, 0.5, "X" + str(i+1), fontsize = 42,
                            horizontalalignment='center', 
                            verticalalignment='center')
        if i < j:
            axes[i,j].xaxis.set_visible(False)
            axes[i,j].yaxis.set_visible(False)
            plt.setp(axes[i,j].spines.values(), visible=False)
        if i > j:
            axes[i,j].scatter(xt[j], xt[i], 
                              edgecolors = ["red" if c == 0 else "black" for c in xt["tr"]], 
                              facecolors = "none")

plt.suptitle("Scatterplot matrix for variables X1 to X7", fontsize = 16)
fig.tight_layout()
plt.show()


#  Plot 5
fig, axes = plt.subplots(7, 7, figsize = (15,15))

for i in range(0, 7):
    for j in range(0, 7):
        if i == j:
            axes[i, j].text(0.5, 0.5, "X" + str(i+8), fontsize = 42, 
                            horizontalalignment='center', 
                            verticalalignment='center')
        if i < j:
            axes[i,j].xaxis.set_visible(False)
            axes[i,j].yaxis.set_visible(False)
            plt.setp(axes[i,j].spines.values(), visible=False)
        if i > j:
            axes[i,j].scatter(xt[j+7], xt[i+7], 
                              edgecolors = ["red" if c == 0 else "black" for c in xt["tr"]], 
                              facecolors = "none")

plt.suptitle("Scatterplot matrix for variables X8 to X14", fontsize = 16)
fig.tight_layout()
plt.show()