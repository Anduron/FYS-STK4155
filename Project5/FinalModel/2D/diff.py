from numpy import *
from matplotlib.pyplot import *

t1 = loadtxt("test.txt")
t2 = loadtxt("test2.txt")
print(len(t1))
u = zeros(len(t1))
u2 = zeros(len(t2))


for j in range(len(t1)):
    for i in range(len(t1)):
        u[j] = t1[j,60]
        u2[j] = t2[j,60]
        if t2[i,j]-t1[i,j] > 0:
            print(t2[i,j]-t1[i,j])



print(u)
x = linspace(0,len(u),len(u))
print(x, len(x))

plot(x,u,u2)
show()
