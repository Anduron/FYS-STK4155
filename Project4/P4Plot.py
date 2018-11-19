import numpy as np
import matplotlib.pyplot as plt
from math import log

infile = open("DL20_T10_MC1M.txt", "r")

infile.readline()

n = infile.readlines()
line = infile.readline()
T = np.zeros(len(n))
E = np.zeros(len(n))
EE = np.zeros(len(n))
M = np.zeros(len(n))
MM = np.zeros(len(n))
Mabs = np.zeros(len(n))
Ev = np.zeros(len(n))
Mv = np.zeros(len(n))

iteration = 0

for line in n:

    T[iteration] = float(line.split()[0])
    #E[iteration] = float(line.split()[1])
    #EE[iteration] = float(line.split()[2])
    #M[iteration] = float(line.split()[3])
    #MM[iteration] = float(line.split()[4])
    #Mabs[iteration] = float(line.split()[5])
    #Ev[iteration] = float(line.split()[6])
    #Mv[iteration] = float(line.split()[7])

    iteration += 1

Cv = Ev/T**2    #(EE - E**2)/(T**2)
Chi = Mv/T    #(MM - M**2)/T


#plt.plot(T , Chi)
#plt.plot(T , Cv)
#plt.plot(T, E)
#plt.plot(T, Mabs)

cycles = np.linspace(1,len(n),len(n));

plt.plot(cycles,T)

plt.xlabel("$T$",size = 15); plt.ylabel("$\\langle E \\rangle$",size=15)
plt.title("",size=15)
plt.legend([""], prop={'size':15})
plt.show()

plt.hist(T[2002:], 48) #119
plt.title("Energy probability distribution of 20x20 lattice",size=15)
plt.xlabel("Energy",size = 15); plt.ylabel("Probability",size=15)
plt.legend(["T=1.0"], prop={'size':15})
plt.show()
