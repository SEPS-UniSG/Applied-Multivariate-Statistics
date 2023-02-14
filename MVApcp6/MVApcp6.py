import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_table("carc.txt", header=None)
df = df.iloc[:,[4,5,6]]

fig, ax = plt.subplots(figsize = (10,10))
bp = plt.boxplot(df, widths = 0.7,
            whiskerprops = {'linestyle': '--', 'linewidth': '2'},
            capprops = {'linewidth': '2'}, capwidths = 0.5,
            boxprops = {'linewidth': '2'}, flierprops = {'linewidth':'2.5', 'color':'red'})
bp['boxes'].set(facecolor = "gray")
plt.xticks([1, 2, 3], ['headroom', 'rear seat', 'trunk space'])
ax.tick_params(axis='both', labelsize=25)
plt.title(label = "Boxplot (Car Data)", 
          fontsize = 30, fontweight = "bold", pad = 15)
