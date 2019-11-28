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


class DNModel(tf.keras.Model):
    def __init__(self):
        super(DNModel, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(50, activation=tf.nn.sigmoid)
        self.out = tf.keras.layers.Dense(3, name="output") #n

    def call(self, inputs):
        x = self.dense_1(inputs)

        return self.out(x)


@tf.function
def rhs(model,A,x,t):
    #print(trial_function(model,x,t).get_shape(),tf.constant(A.get_shape()))
    #r = tf.linalg.matmul(tf.transpose(trial_function(model,x,t)),trial_function(model,x,t))*tf.linalg.matmul(A,trial_function(model,x,t))
    #- tf.linalg.matmul(tf.linalg.matmul(tf.linalg.matmul((tf.transpose(trial_function(model,x,t)),A),trial_function(model,x,t))),trial_function(model,x,t))
    print(tf.linalg.matmul(tf.transpose(trial_function(model,x,t)),trial_function(model,x,t)).get_shape(),A.get_shape(),trial_function(model,x,t).get_shape())
    r = tf.linalg.matmul(tf.transpose(trial_function(model,x,t)),trial_function(model,x,t))*tf.linalg.matmul(A,trial_function(model,x,t))
    - tf.transpose(trial_function(model,x,t))*A*trial_function(model,x,t)*trial_function(model,x,t)
    return r

@tf.function
def trial_function(model,x,t):
    """
    some trial function...
    """
    g = (1-t)*x + t*model(t)
    #g = tf.transpose(g)
    #print(g.get_shape())
    return g


@tf.function
def loss(model, A, x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(t)
        trial = trial_function(model, x, t)

    d_trial_dt = tape.gradient(trial, t)

    del tape

    return tf.losses.MSE(tf.zeros_like(d_trial_dt), d_trial_dt - rhs(model,A,x,t))


@tf.function
def grad(model, A, x, t):
    with tf.GradientTape() as tape:
        #print(A.get_shape(),t.get_shape())
        loss_value = loss(model, A, x, t)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)





def main():
    n = 6

    start = tf.constant(0,dtype=tf.float64)
    stop = tf.constant(9,dtype=tf.float64)
    points = 10

    t = tf.reshape(tf.linspace(start, stop, points),(-1,1))#tf.linspace(start, stop, points)#tf.reshape(tf.linspace(start, stop, points),(-1,1))
    print(t)

    Q = np.random.rand(n,n)

    #A = 0.5*(Q + np.transpose(Q))

    A = tf.constant([[3,2,4],[2,0,2],[4,2,3]], dtype=tf.float64)

    print(f'Suppose a symmetric matrix A = \n{A}')
    print()

    E_val, E_vec = np.linalg.eig(A)

    print(f'The eigenvalues given by the eig function in numpy is E = \n{E_val}')
    print(f'with eigenvectors given by R = \n{E_vec}')

    x = [1,0,0]

    model = DNModel()
    optimizer = tf.keras.optimizers.Adam(0.002)

    num_epochs = 1000
    for t_ in t:
        for epochs in range(num_epochs):
            t_ = tf.reshape(t_,[-1,1])
            cost, gradients = grad(model, A,x,t)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print(f"Step: {optimizer.iterations.numpy()}," + f"Loss: {tf.reduce_mean(cost.numpy())}")

    g_nn = tf.reshape(trial_function(model, x, t), (3, 5))

    NNdiff = tf.abs(E_val - g_nn)
    print(g_nn)
    print(f"NN max diff: {tf.reduce_max(NNdiff)}")
    print(f"NN mean diff: {tf.reduce_mean(NNdiff)}")

    # Run training loop
    # Apply gradients in optimizer
    # Output loss improvement


    # Plot solution on larger grid
    #x = tf.reshape(tf.linspace(start,stop,1001),(-1,1))

if __name__ == "__main__":
    main()
