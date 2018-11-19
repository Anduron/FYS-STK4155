import numpy as np
import matplotlib.pyplot as plt

#files = ["DL20_T10_MC1M.txt", "DL20_T24_MC1M.txt"]
#files = ["OL20_T10_MC1M.txt", "OL20_T24_MC1M.txt"]
#files = ["L40_T20-23_dT0001_MC1M.txt", "L60_T20-23_dT0001_MC1M.txt", "L80_T20-23_dT0001_MC1M.txt", "L100_T20-23_dT0001_MC1M.txt"]
#files = ["L40_T22-24_dT0001_MC1M.txt", "L60_T22-24_dT0001_MC1M.txt", "L80_T22-24_dT0001_MC1M.txt", "L100_T22-24_dT0001_MC1M.txt", "L140_T22-24_dT0001_MC1M.txt"]

DL20 = False
OL20 = False
L40_100 = False

if DL20 == True:
    files = ["DL20_T10_MC1M.txt", "DL20_T24_MC1M.txt"]

    n = len(np.loadtxt("DL20_T10_MC1M.txt",usecols=0))
    cycles = np.linspace(1,n,n);

    for i in files:

        E = np.loadtxt(i, usecols=0)
        M = np.loadtxt(i, usecols=1)

        plt.plot(cycles, abs(M))
        #plt.plot(cycles, E)

    plt.title("Magnetization with varying temperatures and L=20",size=15)
    plt.xlabel("cycles",size=15); plt.ylabel("$\\langle |M| \\rangle$/L^2",size=15)
    plt.legend(["1.0", "2.4"],prop={"size": 15})

    plt.show()

if OL20 == True:
    files = ["OL20_T10_MC1M.txt", "OL20_T24_MC1M.txt"]

    n = len(np.loadtxt("OL20_T10_MC1M.txt",usecols=0))
    cycles = np.linspace(1,n,n);

    for i in files:

        E = np.loadtxt(i, usecols=0)
        M = np.loadtxt(i, usecols=1)

        plt.plot(cycles, abs(M))
        #plt.plot(cycles, E)

    plt.title("Magnetization with varying temperatures and L=20",size=15)
    plt.xlabel("cycles",size=15); plt.ylabel("$\\langle |M| \\rangle$/L^2",size=15)
    plt.legend(["1.0", "2.4"],prop={"size": 15})

    plt.show()


if L40_100 == True:
    files = ["L40_T20-23_dT0001_MC1M.txt", "L60_T20-23_dT0001_MC1M.txt", "L80_T20-23_dT0001_MC1M.txt", "L100_T20-23_dT0001_MC1M.txt"]
    #files = ["L40_T22-24_dT0001_MC1M.txt", "L60_T22-24_dT0001_MC1M.txt", "L80_T22-24_dT0001_MC1M.txt", "L100_T22-24_dT0001_MC1M.txt", "L140_T22-24_dT0001_MC1M.txt"]

    for i in files:

        T = np.loadtxt(i, usecols=0)
        E = np.loadtxt(i, usecols=1)
        M = np.loadtxt(i, usecols=5)
        Cv = np.loadtxt(i, usecols=6)
        Chi = np.loadtxt(i, usecols=7)


        #plt.plot(T,E)
        #plt.figure()
        #plt.plot(T,M)
        #plt.figure()
        #plt.plot(T,Cv)
        #plt.figure()
        plt.plot(T,Chi)

    plt.title("Susceptibilities with varying lattice size",size=15)
    plt.xlabel("T[kT/J]",size=15); plt.ylabel("$\\chi$/L^2",size=15)
    plt.legend(["L=40", "L=60", "L=80", "L=100"],prop={"size": 15})

    plt.show()
