import numpy as np

n = 2
T = 1.0

E = -8.0*np.sinh(8/T)/(3.0 + np.cosh(8/T))
Cv = 1.0/T**2*(64.0*np.cosh(8/T)*(3 + np.cosh(8/T)) - 64*np.sinh(8/T)**2)/(3+np.cosh(8/T))**2
Mabs = (2*np.exp(8/T) + 4.0)/(3.0 + np.cosh(8/T))
M2 = 8.0*(np.exp(8/T) + 1.0)/(3.0 + np.cosh(8/T))
Chi = M2/T

print(E/n**2, Mabs/n**2, Cv/n**2, Chi/n**2)
