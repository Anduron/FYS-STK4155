import numpy as np
import matplotlib.pyplot as plt


x = np.linspace(-3,3,50)
y = x**2
y2 = 0.5*x**2

plt.plot(x,y)
plt.plot(x,y2)
plt.xlabel("x",size = 15); plt.ylabel("y",size=15)
plt.title("Quadratic potential",size=15)
plt.axhline(y=0, color = "black")
plt.axvline(x=0, color = "black")
plt.legend(["$\\omega_{r} = 1$","$\\omega_{r} = 0.5$"], prop={'size':15}, loc = 'upper right')
plt.show()
