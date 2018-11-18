import numpy as np
import matplotlib.pyplot as plt

#files = ["L40_T20-23_dT0001_MC1M.txt", "L60_T20-23_dT0001_MC1M.txt", "L80_T20-23_dT0001_MC1M.txt", "L100_T20-23_dT0001_MC1M.txt"]
files = ["L40_T22-24_dT0001_MC1M.txt", "L60_T22-24_dT0001_MC1M.txt", "L80_T22-24_dT0001_MC1M.txt", "L100_T22-24_dT0001_MC1M.txt", "L140_T22-24_dT0001_MC1M.txt"]


for i in files:

    T = np.loadtxt(i, usecols=0)
    E = np.loadtxt(i, usecols=1)
    M = np.loadtxt(i, usecols=5)
    Cv = np.loadtxt(i, usecols=6)
    Chi = np.loadtxt(i, usecols=7)

    #plt.plot(T,E)

    #plt.figure()
    plt.plot(T,M)
    #plt.figure()
    #plt.plot(T,Cv)
    #plt.figure()
    #plt.plot(T,Chi)

plt.title("",size=15)
plt.xlabel("",size=15); plt.ylabel("",size=15)
plt.legend(["L=40", "L=60", "L=80", "L=100", "L=140"],prop={"size": 15})

plt.show()
