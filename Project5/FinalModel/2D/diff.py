from numpy import *
from matplotlib.pyplot import *

t1 = loadtxt("5NQNslab.txt")
t2 = loadtxt("5YQNslab.txt")
t3 = loadtxt("5YQYslab.txt")

u1 = zeros(len(t1))
u2 = zeros(len(t2))
u3 = zeros(len(t3))

for j in range(len(t1)):
    for i in range(len(t1)):
        u1[j] = t1[j,60]
        u2[j] = t2[j,60]
        u3[j] = t3[j,60]


x = linspace(0,len(u1),len(u1))

plot(x,u1,u2)
plot(x,u3)
grid(True)
title('Temperature distribution in center of lithosphere \n at t = 1 GYr as function of depth',size=15)
legend(["No Q","Q","Enriched Q"], prop = {'size': 15})
xlabel('$y$', size=15); ylabel('$u(y)$', size=15)
show()
