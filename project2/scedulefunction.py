import matplotlib.pyplot as plt
import numpy as np

t0 = 1
t1 = 10

t = np.linspace(0,t0/t1,1000)

def schedule(t, t0,t1):
    return t0/(t1+t)

slope = t**3

def schedule2(t, t0,t1,slope):
    return t0/(t1+t)*slope

#sch = schedule(t,t0,t1)
sch2 = schedule2(t,t0,t1,slope)

#plt.plot((t),sch)
plt.plot(t,sch2)

plt.show()
