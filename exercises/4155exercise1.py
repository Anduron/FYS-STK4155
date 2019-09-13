import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as skl
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

x = np.random.rand(100, 1)
y = 5*x*x +0.1*np.random.randn(100, 1)

poly = PolynomialFeatures(degree = 2)



plt.plot(x,y, ".")
plt.show()
