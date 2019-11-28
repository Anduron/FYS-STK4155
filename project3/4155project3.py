import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn.linear_model as skl
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split, KFold
import scipy.linalg as scl
import pandas as pd
from imageio import imread
import tensorflow as tf
import sys

def FDM_diffusion_solver(x,dx,t,dt,I,plot=False):
    """
    Solves a diffusion equation using a finite differences scheme.
    IMPORTANT NOTE: Assuming Dirichlet boundary conditions and no source term!
    #print(np.linalg.norm(u[n+1,:]))
    """

    u = np.zeros((len(t),len(x)))
    u[0,:] = I

    if plot==False:

        for n in range(0,len(t)-1):
            sys.stdout.write("\rCompletion: %d %%" %(100*(n+1)/len(t)))
            sys.stdout.flush()

            u[n+1,1:-1] = u[n,1:-1] + (dt/(dx**2))*(u[n,2:] - 2*u[n,1:-1] + u[n,:-2])
            u[n+1,0] = 0; u[n+1,-1] = 0
        print()

    else:

        fig = plt.figure(111)
        ax = fig.add_subplot(111)

        for n in range(0,len(t)-1):

            u[n+1,1:-1] = u[n,1:-1] + (dt/(dx**2))*(u[n,2:] - 2*u[n,1:-1] + u[n,:-2])
            u[n+1,0] = 0; u[n+1,-1] = 0

            ax.clear()
            ax.set_ylim(-1.5,1.5)
            ax.plot(x, u[n+1,:], 'r')
            ax.set_title("step %s" %(n))
            plt.pause(0.0001)

    return u

def initial_condition(x):
    """
    Takes variable or array x.
    Returns a variable or array of a chosen initial condtion.
    """
    return np.sin(np.pi*x)


def u_analytic(x,t):
    """
    Analytically found solution of du/dt = d^2(u(x,t))/dx^2
    for x = [0,1] and I(x) = sin(pi x), Simplest case of heat equation.
    """
    u = np.exp(-t*np.pi**2)*np.sin(np.pi*x)
    return u


def trial_function(x,t):
    """
    some trial function...
    """
    g1 = (1-t)*initial_condition(x)
    g2 = x*(x-1)*t
    return g1, g2


def main():

    #Setting up the domain of the problem
    L = 1
    T = 1

    #Stability criterion Beta <= 1
    Beta = 0.8

    dx = 0.025 #0.1
    dt = 0.5*Beta*dx*dx

    Nx = int(L/dx)
    Nt = int(T/dt)

    x = np.linspace(0,L,Nx)
    t = np.linspace(0,T,Nt)

    print(f"The domain is: T = {T} with {Nt} points and L = {L} with {Nx} points")
    print(f"Stability coefficient is Beta = {Beta}, with dx = {dx}, and dt = {dt}")

    #Setting up and showing the initial condition.
    I = np.zeros(len(x))
    I = initial_condition(x)
    #I[0] = 0; I[-1] = 0 #Enforcing boundaries

    #plt.plot(x,I)
    #plt.show()

    #Calling the FDM solver
    u = FDM_diffusion_solver(x,dx,t,dt,I)

    #Calling the Neural Network


    #plotting the time evolution of our solvers
    fig = plt.figure(111)
    ax = fig.add_subplot(111)

    for n in range(Nt-1):

        ax.clear()
        ax.set_ylim(-1.5,1.5)
        ax.plot(x, u[n+1,:], 'r')
        ax.plot(x, u_analytic(x,dt*n),'c')
        ax.set_title("step %s" %(n))
        plt.legend(["numerical","analytic"])
        plt.pause(0.0001)

if __name__ == "__main__":
    main()
