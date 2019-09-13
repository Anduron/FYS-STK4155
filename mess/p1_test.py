import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn.linear_model as skl
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd


def FrankeFunction(x,y,n,noise):
    epsilon = np.random.normal(0,noise,n*n).reshape(n,n)

    t1 = 0.75*np.exp(-((9*x-2)**2)/4 - ((9*y-2)**2)/4)
    t2 = 0.75*np.exp(-((9*x+1)**2)/49 - ((9*y+1)**2)/10)
    t3 = 0.5*np.exp(-((9*x-7)**2)/4 - ((9*y-3)**2)/4)
    t4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    f = t1 + t2 + t3 + t4 + epsilon
    return f


def X_DesignMatrix(x,y,n,degree = 5):

    if len(x.shape) > 1:
	       x = np.ravel(x)
	       y = np.ravel(y)

    lB = int((degree+1)*(degree+2)/2)
    X = np.ones((n,lB))

    for i in range(1,degree+1):
        j = int(i*(i+1)/2)
        for k in range(i+1):
            X[:, j+k] = (x**(i-k))*(y**k)
    return X



def plotting_function(x,y,z):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(x,y,z,cmap=cm.coolwarm,linewidth=0,antialiased = False)

    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    return fig.colorbar(surf, shrink=0.5, aspect=5)


def OLS(X,n,y_tilde):
    B = np.linalg.inv(np.dot(X.T,X)).dot(X.T).dot(y_tilde)
    z_predict = np.dot(X,B).reshape(n,n)
    varB = np.trace(np.linalg.inv(np.dot(X.T,X)))
    return z_predict, varB

def ridge(X,n,lmd,ytilde):
    B = np.linalg.inv(np.dot(X.T,X)).dot(X.T).dot(y_tilde)
    z_predict = np.dot(X,B) #.reshape(n,n)
    varB = np.trace(np.linalg.inv(np.dot(X.T,X)))
    return z_predict, varB

def kCrossValidation(x,y,z,deg_max,k=5):
    kfold = KFold(n_splits = k)
    r2_scores = np.zeros((deg_max,k))
    i = 0
    for deg in range(0,deg_max):

        j = 0
        for tr_idx, ts_idx in kfold.split(x):
            x_train = x[tr_idx]
            y_train = y[tr_idx]
            z_train = z[tr_idx]

            x_test = x[ts_idx]
            y_test = y[ts_idx]
            z_test = z[ts_idx]

            X_DM = X_DesignMatrix(x,y,len(x_train),deg)
            z_predict = OLS(X_DM,int(np.sqrt(len(x_train))),z_train)

            r2_scores[i,j] = metrics.r2_score(z_train,z_predict)

            j += 1
        i += 1

    return r2_scores

def main():
    np.random.seed(0)


    n = 1000
    noise = 0.03
    test_S = 0.3
    degree = 5
    k = 5
    deg_max = 10

    x = np.sort(np.random.uniform(0,1,n))
    y = np.sort(np.random.uniform(0,1,n))
    x,y = np.meshgrid(x,y)
    x_1 = np.ravel(x)
    y_1 = np.ravel(y)

    z = FrankeFunction(x,y,n,noise)
    z_1 = np.ravel(z)

    X_DM = X_DesignMatrix(x_1,y_1,n*n,degree)

    z_predict, varB = OLS(X_DM,n,z_1)

    r2_score = metrics.r2_score(z,z_predict)
    MSE = metrics.mean_squared_error(z,z_predict)
    print("MSE = %s, R2 score = %s, Variance B = %s" %(MSE,r2_score,varB))

    plotting_function(x,y,z)
    plotting_function(x,y,z_predict)

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x_1,y_1,z_1,test_size=test_S)
    r2_scores = kCrossValidation(x_train,y_train,z_train,deg_max,k)

if __name__ == "__main__":
    main()
    plt.show()
