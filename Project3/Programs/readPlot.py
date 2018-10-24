import numpy as np
import matplotlib.pyplot as plt
from math import log

infile = open("test.txt", "r")

n = infile.readlines()
line = infile.readline()
x = np.zeros(len(n))
y = np.zeros(len(n))
x2 = np.zeros(len(n))
y2 = np.zeros(len(n))
x3 = np.zeros(len(n))
y3 = np.zeros(len(n))

iteration = 0

for line in n:

    x[iteration] = line.split()[0]
    y[iteration] = line.split()[1]
    x2[iteration] = line.split()[2]
    y2[iteration] = line.split()[3]
    x3[iteration] = line.split()[4]
    y3[iteration] = line.split()[5]
    iteration += 1


plt.plot(x , y)
plt.plot(x2,y2)
plt.plot(x3,y3)

#plt.xlabel("$\log(n)$",size = 15); plt.ylabel("$\log(iterations)$",size=15)
plt.title("Iterations as function of matrix size n",size=15)
#plt.legend(["$\log_{10}(iterations(n))$", "$y = 2\log_{10}(n)$"], prop={'size':15})
plt.show()
