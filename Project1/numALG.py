import numpy as np
import matplotlib.pyplot as plt

def algfunc(a,b,c,n):

    avec = np.zeros(n+1); avec[0] = a
    bvec = np.zeros(n+1); bvec[0] = b
    cvec = np.zeros(n+1); cvec[0] = c

    btld = np.zeros(n+1); btld[0] = b
    f = np.zeros(n+1)
    v = np.zeros(n+1)
    ftld = np.zeros(n+1); ftld[0] = f[0]


    h = 1.0/float(n)


    #btd = f*h**2
    for i in range(1, n):
        btld[i] = bvec[i] - ((avec[i-1]*cvec[i-1])/(btld[i-1]))
        ftld[i] = f[i] - ((ftld[i-1]*avec[i-1])/(btld[i-1]))

    v[n] = ftld[n]/btld[n]

    for i in range(n,1):
        v[i] = (ftld[i]-f[i]*u[i+1])/(btld[i])

    return v

u = algfunc(-1,2,-1,10)
print(u)
#x = np.linspace(0,1,10)
#plt.plot(x,u)
#plt.show()
