import numpy as np

a = np.linspace(0,31,32).reshape(4,8)
print(a)
b = np.ravel(a)
print(b)
c = b.reshape(8,4)
print(c)
d = c.reshape(4,8)
print(d)
