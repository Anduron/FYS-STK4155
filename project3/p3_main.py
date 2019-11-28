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
import math
import sys
import os


tf.keras.backend.set_floatx("float64")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
#np.random.seed(42)

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


class DNModel(tf.keras.Model):
    def __init__(self):
        super(DNModel, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(60, activation=tf.nn.sigmoid)
        self.dense_2 = tf.keras.layers.Dense(30, activation=tf.nn.sigmoid)
        self.out = tf.keras.layers.Dense(1, name="output")

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dense_2(x)

        return self.out(x)


@tf.function
def initial_condition(x):
    """
    Takes variable or array x.
    Returns a variable or array of a chosen initial condtion.
    """
    return tf.sin(math.pi*x)


def u_analytic(x,t):
    """
    Analytically found solution of du/dt = d^2(u(x,t))/dx^2
    for x = [0,1] and I(x) = sin(pi x), Simplest case of heat equation.
    """
    u = np.exp(-t*np.pi**2)*np.sin(np.pi*x)
    return u


@tf.function
def trial_function(model, x,t):
    """
    some trial function...
    """
    points = tf.concat([x, t], axis=1)

    g = (1-t)*initial_condition(x) + x*(x-1)*t*model(points)

    return g


@tf.function
def loss(model, x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x,t])
        with tf.GradientTape(persistent=True) as tape_2:
            tape_2.watch([x,t])

            trial = trial_function(model, x, t)

        d_trial_dx = tape_2.gradient(trial, x)
        d_trial_dt = tape_2.gradient(trial, t)

    d2_trial_d2x = tape.gradient(d_trial_dx, x)

    del tape_2
    del tape

    return tf.losses.MSE(tf.zeros_like(d2_trial_d2x), d2_trial_d2x - d_trial_dt)


@tf.function
def grad(model, x, t):
    with tf.GradientTape() as tape:
        loss_value = loss(model, x, t)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


def plot_heat(u,f,g,x,t,ptype=1):
    if ptype == 1:
        #plotting the time evolution of our solvers
        fig = plt.figure(111)
        ax = fig.add_subplot(111)

        Nt = len(t)
        for n in range(Nt-1):

            ax.clear()
            ax.set_ylim(-1.5,1.5)
            ax.plot(x, f[n,:], 'g')
            ax.plot(x, u[n,:],'c')
            ax.plot(x, g[n,:],'r')
            ax.text(0.3,-0.3,f"NN max diff   = {np.max(np.abs(u[n,:] - g[n,:])):.9f}")
            ax.text(0.3,-0.4,f"FDM max diff = {np.max(np.abs(u[n,:] - f[n,:])):.9f}")
            ax.set_title("step %s" %(n))
            plt.legend(["Analytic","FDM scheme", "DNN model"])
            plt.pause(0.0001)

    elif ptype == 2:
        fig1 = plt.figure()
        ax = fig1.gca(projection="3d")
        ax.set_title("u")
        ax.plot_surface(t, x, u)

        fig2 = plt.figure()
        ax = fig2.gca(projection="3d")
        ax.set_title("g")
        ax.plot_surface(t, x, g)

        diff = tf.abs(u - g)
        fig3 = plt.figure()
        ax = fig3.gca(projection="3d")
        ax.set_title("Diff")
        ax.plot_surface(t, x, diff)
        plt.show()


def main():

    #Setting up the domain of the problem
    L = tf.constant(1,dtype=tf.float64)
    T = tf.constant(1,dtype=tf.float64)

    #Stability criterion Beta <= 1
    Beta = 0.8

    dx = 0.1 #0.1
    dt = 0.5*Beta*dx*dx

    Nx = int(L/dx)
    Nt = int(T/dt)

    x = tf.linspace(tf.constant(0,dtype=tf.float64),L,Nx)
    t = tf.linspace(tf.constant(0,dtype=tf.float64),T,Nt)

    Xt, Tt = tf.meshgrid(tf.linspace(tf.constant(0,dtype=tf.float64),L,21),tf.linspace(tf.constant(0,dtype=tf.float64),T,21))
    xt, tt = tf.reshape(Xt,[-1,1]), tf.reshape(Tt,[-1,1])

    print(f"The domain is: T = {T} with {Nt} points and L = {L} with {Nx} points")
    print(f"Stability coefficient is Beta = {Beta}, with dx = {dx}, and dt = {dt}")

    #Setting up and showing the initial condition.
    I = np.zeros(len(x))
    I = initial_condition(x)
    #I[0] = 0; I[-1] = 0 #Enforcing boundaries

    #plt.plot(x,I)
    #plt.show()

    #precalculating analytic solution
    u = np.zeros((Nt,Nx))
    for n in range(Nt-1):
        u[n,:] = u_analytic(x,n*dt)

    #Calling the FDM solver
    u_FDM = FDM_diffusion_solver(x,dx,t,dt,I)

    #Calling the Neural Network
    model = DNModel()
    optimizer = tf.keras.optimizers.Adam(0.01)

    num_epochs = 1000

    for epoch in range(num_epochs):
        cost, gradients = grad(model, xt, tt)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        print(
            f"Step: {optimizer.iterations.numpy()}, "
            + f"Loss: {tf.math.reduce_mean(cost.numpy())}"
        )

    #num_points = Nx
    Xt, Tt = tf.meshgrid(
        tf.linspace(tf.constant(0,dtype=tf.float64),tf.constant(1,dtype=tf.float64), Nx),
        tf.linspace(tf.constant(0,dtype=tf.float64),tf.constant(1,dtype=tf.float64), Nt)
    )
    xt, tt = tf.reshape(Xt, [-1, 1]), tf.reshape(Tt, [-1, 1])

    g_nn = tf.reshape(trial_function(model, xt, tt), (Nt, Nx))


    FDMdiff = np.abs(u - u_FDM)
    print(f"FDM max diff: {tf.reduce_max(FDMdiff)}")
    print(f"FDM mean diff: {tf.reduce_mean(FDMdiff)}")

    NNdiff = tf.abs(u - g_nn)
    print(f"NN max diff: {tf.reduce_max(NNdiff)}")
    print(f"NN mean diff: {tf.reduce_mean(NNdiff)}")

    plot_heat(u,u_FDM,g_nn,Xt,Tt,ptype=2)
    plot_heat(u,u_FDM,g_nn,x,t)


if __name__ == "__main__":
    main()
