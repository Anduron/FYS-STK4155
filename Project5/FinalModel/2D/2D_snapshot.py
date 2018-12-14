import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation

fe_u = np.loadtxt('1GY_noQ.txt')
be_u = np.loadtxt('1GY_Q_noSlab.txt')
an_u = np.loadtxt('1GY_Q_slab.txt')
u = fe_u
#u = an_u

nx = len(u[0,:])
nt = int(len(u[:,0])/len(u[0,:]))
x = np.linspace(0, 1, nx)
dx = 1./nx
print(nx, nt)
#t_steps = int(len(u[:,0])/len(u[0,:]))  # get number of time steps from matrix
#t_steps = int(nt/nx)

mat = np.zeros((nt, nx, nx))
# Create matrices for individual time steps
for t in range(nt):
    start = nx*t
    end = start+nx
    mat[t] = u[start:end]

# Plot specific points in time
#index = 0
#t = float(index)/float(nt)
t = 0.99
index = int(t*nt)
print (index, t)
fig = plt.figure(figsize=(8,8))
im = plt.imshow(mat[index], cmap=cm.coolwarm)
cbar = plt.colorbar()
plt.clim(0,1)
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('$\\Delta T$', rotation=270, size=15)
plt.xlabel('$x$', size=15); plt.ylabel('$y$', size=15)
plt.title('Temperature distribution at t = %1.2f GYr\n with no radioactive perturbations' % (t), size=15)
#plt.title('Temperature distribution at t = %1.2f\nin a %s x %s grid (analytic)' % (t,nx,nx), size=15)
#plt.savefig('figures/fe_error_2D_dx=%s_t=%1.2f.eps' % (dx,t))
plt.savefig('1GY_noQ.pdf')

plt.show()
