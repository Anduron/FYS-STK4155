import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg

file1 = "L40_T22-24_dT0001_MC1M.txt"
file2 = "L60_T22-24_dT0001_MC1M.txt"
file3 = "L80_T22-24_dT0001_MC1M.txt"
file4 = "L100_T22-24_dT0001_MC1M.txt"
file5 = "L140_T22-24_dT0001_MC1M.txt"

files = [file1, file2, file3, file4, file5]

chi1, T = np.loadtxt(file1, usecols=(7,0), unpack=True)
chi2 = np.loadtxt(file2, usecols=7)
chi3 = np.loadtxt(file3, usecols=7)
chi4 = np.loadtxt(file4, usecols=7)
chi5 = np.loadtxt(file5, usecols=7)

L = np.array([1.0/140, 1.0/100, 1.0/80, 1.0/60, 1.0/40])
T = np.array([T[np.argmax(chi5)]-L[0], T[np.argmax(chi4)] - L[1], T[np.argmax(chi3)]- L[2], T[np.argmax(chi2)]- L[3], T[np.argmax(chi1)]- L[4]])

z = np.polyfit(L,T,1)
zz = np.poly1d(z)

print(zz[0], zz[1])

plt.plot(L, T,'o')
plt.plot(L, zz(L))
plt.legend(['Data', 'Fitted'], prop={'size':15})
plt.title("Linear fitting of critical temperature as a function of L",size=15)
plt.xlabel('1/L', size=15); plt.ylabel('$T_C$ [k/J]')


plt.show()
