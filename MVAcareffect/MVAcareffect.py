import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pyreadr

data = pyreadr.read_r("carc.rda")
carc = data['carc']
#carc = pd.read_table("carc.dat", header = None, sep = "\s+")
carc = carc[["M", "W", "D", "C", "P"]]
carc.columns = ["Mileage", "Weight", "Displacement", "Origin", "Price"]

# Reasonable model
lm1 = ols(formula="np.log(Mileage) ~ np.log(Weight) + np.log(Displacement) + Origin", data=carc).fit()
#print(lm1.summary())

# Model without Origin
lm2 = ols(formula="np.log(Mileage) ~ np.log(Weight) + np.log(Displacement)", data=carc).fit()
#print(lm2.summary())

# Test whether Origin is significant
anova_results = sm.stats.anova_lm(lm1, lm2)
print(anova_results)

# Plot lm1
fig1 = plt.figure()
sm.graphics.plot_partregress_grid(lm1, fig=fig1)

# Model with Weight and Origin
lm3 = ols(formula="np.log(Mileage) ~ np.log(Weight) + Origin", data=carc).fit()
#print(lm3.summary())

# Plot log(Mileage) vs log(Weight) with different colors for Origin
fig2, ax = plt.subplots()
colors = np.where(carc["Origin"] == "US", 2, np.where(carc["Origin"] == "Japan", 3, 4))
ax.scatter(np.log(carc["Weight"]), np.log(carc["Mileage"]), c=colors, cmap="viridis")
weights_ordered = carc["Weight"].argsort()
ax.plot(np.log(carc["Weight"])[weights_ordered], lm3.predict(carc.iloc[weights_ordered]), color="red", label="US")
ax.plot(np.log(carc["Weight"])[weights_ordered], lm3.predict(carc.iloc[weights_ordered]) + lm3.params[2], color="green", linestyle="--", label="Japan")
ax.plot(np.log(carc["Weight"])[weights_ordered], lm3.predict(carc.iloc[weights_ordered]) + lm3.params[3], color="blue", linestyle="--", label="Europe")
ax.legend()

plt.show()
