import numpy as np
import matplotlib.pyplot as plt
from math import log

list = ["r2e4.txt", "r2e1.txt", "r2e2.txt", "r2e3.txt"]
for file in list:

    infile = open(file,"r")
    n = infile.readlines()
    line = infile.readline()
    x = np.zeros(len(n))
    u = np.zeros(len(n))
    error = np.zeros(len(n))

    iteration = 0


    for line in n:

        x[iteration] = line.split()[0]
        #x[iteration] = log(x[iteration])
        u[iteration] = line.split()[1]
        error[iteration] = line.split()[2]
        #error[iteration] = log(error[iteration])
        iteration += 1


    plt.plot(x , error)

#x2 = np.linspace(1,6,len(n))
#y2 = 2*x2

#plt.plot(x2,y2, "--")
plt.xlabel("$\\rho_{n}-\\rho$", size=15); plt.ylabel("Probability", size=15)
plt.title("Probability distribution", size=15)
plt.legend(["No Coulomb interaction", "$\\omega_{r} = 0.05$", "$\\omega_{r} = 0.5$", "$\\omega_{r} = 1$"], prop={'size': 15})
plt.show()
