import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn.linear_model as skl
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import StandardScaler
import scipy.linalg as scl
import pandas as pd
from imageio import imread
import seaborn as sns
import sys
from NN import *

plt.rcParams.update({'font.size': 12})

def FrankeFunction(x,y):
    """
    Creates a franke function with or without noise.
    """

    t1 = 0.75*np.exp(-((9*x-2)**2)/4 - ((9*y-2)**2)/4)
    t2 = 0.75*np.exp(-((9*x+1)**2)/49 - ((9*y+1)**2)/10)
    t3 = 0.5*np.exp(-((9*x-7)**2)/4 - ((9*y-3)**2)/4)
    t4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    f = t1 + t2 + t3 + t4
    return f

def Franke_dataset(n, noise=0.5):
    # Generate dataset from Franke function with given noise
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)

    # Create X, z
    X = np.zeros((n*n, 2))
    z = np.zeros(n*n)
    print (X.shape)
    eps = np.asarray([np.random.normal(0,noise,n*n)])
    eps = np.reshape(eps, (n,n))

    for i in range(n):
        for j in range(n):
            X[i*n + j] = [x[i], y[j]]
            z[i*n + j] = FrankeFunction(x[i],y[j]) + eps[i,j]

    x, y = np.meshgrid(x,y)
    z = z/np.max(z)

    return X, x, y, z


def plotting_function(x,y,z,n):
    """
    Plots a 3d surface.
    """

    z = np.reshape(z,(n,n))
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(x,y,z,cmap=cm.coolwarm,linewidth=0,antialiased = False)

    ax.set_zlim(np.min(z), np.max(z))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    return fig.colorbar(surf, shrink=0.5, aspect=5)


def main():
    """
    Performs Data analysis on the Franke function with noise, using OLS, Ridge
    and Lasso with crossvalidation.
    """
    np.random.seed(42) #The meaning of live

    n = 50

    #The dataset
    X, x, y, z = Franke_dataset(n,0.1)#,0)
    print(X)
    #X = StandardScaler(X); y = StandardScaler(y); y = StandardScaler(y); z = StandardScaler(z)
    X_true, x_true, y_true, z_true = Franke_dataset(n,0)
    #X_true = StandardScaler(X); x_true = StandardScaler(x_true); y_true = StandardScaler(y_true); z_true = StandardScaler(z)

    plotting_function(x,y,z,n)
    plt.show()

    X_train, X_test, Z_train, Z_test = train_test_split(X,z,test_size=0.2)
    #X_TRtrue, X_TEtrue, Z_TRtrue, Z_TEtrue = train_test_split(X_true,z_true,test_size=0.3)

    print(len(X_train)+len(X_test))

    epochs = 40
    batch_size = 100
    hidden_neurons = 100
    categories = 10
    max_iter = 1000
    error_min = 0.0001
    eta_vals = np.logspace(-4, -2, 10) #np.linspace(0.0015,0.0025,10)
    lmbd = 0
    #alpha_vals = np.logspace(-5, 1, 7)
    # store the models for later use
    DNN_numpy = np.zeros((len(eta_vals)), dtype=object)
    print(len(X_train), len(Z_train))
    layers = [2, 100,20,5, 1]
    activations = ["tanh","tanh","sigmoid","linear"]#,"sigmoid","sigmoid","softmax"]
    #print(len(activations))

    # grid search
    for i, eta in enumerate(eta_vals):
        alpha = 0.1
        dnn = NeuralNetwork(X_train,Z_train.reshape((len(Z_train),1)),layers, activations,
        epochs=epochs, batches=batch_size, max_iter=max_iter,
        error_min=error_min, eta=eta, lmbd=lmbd, alpha=alpha)
        """
        dnn = NeuralNetwork(X_data=X_train, Y_data=Y_train_onehot,
        hidden_neurons=hidden_neurons, categories=categories,
        epochs=epochs, batches=batch_size, max_iter=max_iter,
        error_min=error_min, eta=eta, lmbd=lmbd, alpha=alpha)
        """
        """
        dnn.train()
        test_predict = dnn.predict(X_test)
        print(np.shape(X_test))
        """
        #dnn = NeuralNetwork(X_train, Y_train_onehot)#, hidden_neurons=hidden_neurons, categories=categories, learningrate=learningrate, lmbd=lmbd, epochs=epochs, batches=batches, max_iter=max_iter, error_min=error_min)
        dnn.SGD()
        test_predict = dnn.predict(X)#.reshape((len(X_test),1)))
        DNN_numpy[i] = dnn

        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("R2 score on test set: ", metrics.r2_score(z, test_predict))
        print("MSE on test set: ", metrics.mean_squared_error(z, test_predict))
        print()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(x, y, test_predict.reshape((n,n)))
        ax.set_title("Prediction of Franke's function")

        plt.show()

    sns.set()

    train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

    for i in range(len(eta_vals)):
        dnn = DNN_numpy[i]
        train_pred = dnn.predict(X_train)
        test_pred = dnn.predict(X_test)

        train_accuracy[i] = metrics.r2_score(Y_train, train_pred)
        test_accuracy[i] = metrics.r2_score(Y_test, test_pred)


    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(train_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Training Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

    fig, ax = plt.subplots(figsize = (10, 10))
    sns.heatmap(test_accuracy, annot=True, ax=ax, cmap="viridis")
    ax.set_title("Test Accuracy")
    ax.set_ylabel("$\eta$")
    ax.set_xlabel("$\lambda$")
    plt.show()

    #self, X_in, y_in, hidden_neurons=20, categories=10, learningrate=0.1, lmb=0, epochs=5, batches=50, max_iter=1e3, error_min=0
    #dnn.SGD()
    #dnn = NeuralNetwork(X_train, Y_train_onehot)#, hidden_neurons=hidden_neurons, categories=categories, learningrate=learningrate, lmbd=lmbd, epochs=epochs, batches=batches, max_iter=max_iter, error_min=error_min)
    #test_predict, prediction = dnn.predictions(X_test)
    #print(prediction)


if __name__== "__main__":
    main()
