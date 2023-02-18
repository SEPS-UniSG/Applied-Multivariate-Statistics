import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('bostonh.dat', sep='\s+', header=None, names=np.arange(1,15,1))

df["name"] = df[14]
df.loc[df["name"] <= df["name"].median(), ["name"]] = 1
df.loc[df["name"] > df["name"].median(), ["name"]] = 2
data = [df.loc[df["name"] == 1,[1]], df.loc[df["name"] == 2,[1]]]

fig, axs = plt.subplots(6,6, figsize=(20,20))
axs[0,0].boxplot(data)
