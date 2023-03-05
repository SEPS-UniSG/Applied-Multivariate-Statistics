# works on pandas 1.5.2, numpy 1.23.5, scikit-learn 1.2.0 and matplotlib 3.6.2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

x = pd.read_csv("pullover.dat", sep = "\s+", header=None)

x1 = (x.iloc[:,1].to_numpy()).reshape((-1, 1))
y = x.iloc[:,0].to_numpy()
reg = LinearRegression().fit(x1, y)
xfit = (np.linspace(70,130,400)).reshape((-1, 1))
yfit = reg.predict(xfit)

fig, ax = plt.subplots(figsize = (10, 10))
ax.scatter(x.iloc[:,1], x.iloc[:,0], c = "w", edgecolors = "black")
ax.plot(xfit,yfit, c = 'r')
plt.xlim(78, 127)
plt.ylim(80, 240)
plt.xlabel("Price (X2)", fontsize = 14)
plt.ylabel("Sales (X1)", fontsize = 14)
plt.title("Pullovers Data", fontsize = 14)
plt.show()


r_sq = reg.score(x1, y)
print('coefficient of determination:', r_sq)
print('intercept:', reg.intercept_)
print('slope:', reg.coef_) 