import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn.linear_model as skl
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures
import scipy.linalg as scl
import pandas as pd
from imageio import imread


def FrankeFunction(x,y,n,noise,noisy=True):
    epsilon = np.random.normal(0,noise,n*n).reshape(n,n)

    t1 = 0.75*np.exp(-((9*x-2)**2)/4 - ((9*y-2)**2)/4)
    t2 = 0.75*np.exp(-((9*x+1)**2)/49 - ((9*y+1)**2)/10)
    t3 = 0.5*np.exp(-((9*x-7)**2)/4 - ((9*y-3)**2)/4)
    t4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    if noisy:
        f = t1 + t2 + t3 + t4 + epsilon
    else:
        f = t1+t2+t3+t4
    return f

def DataImport(filename):
    # Load the terrain
    terrain1 = imread(filename)
    # Show the terrain
    plt.figure()
    plt.imshow(terrain1, cmap='gray')

    #return x, y, z


def X_DesignMatrix(x,y,degree = 5):
    n = len(x)
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



def plotting_function(x,y,z,n):
    z = np.reshape(z,(n,n))
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(x,y,z,cmap=cm.coolwarm,linewidth=0,antialiased = False)

    ax.set_zlim(np.min(z), np.max(z))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    return fig.colorbar(surf, shrink=0.5, aspect=5)


def Coeff(X,y_tilde):
    B = np.linalg.inv(np.dot(X.T,X)).dot(X.T).dot(y_tilde)

    #U, S, V = scl.svd(X)
    #B = V.T @ scl.pinv(scl.diagsvd(S,U.shape[0], V.shape[0])) @ U.T @ y_tilde
    return B

def CoeffRidge(X,y_tilde, lmb):
    B = np.linalg.inv(np.dot(X.T,X) + lmb*np.identity(len(np.dot(X.T,X)))).dot(X.T).dot(y_tilde)
    return B


def kCrossValidation(x,y,z,deg_max,k=5):

    kfold = KFold(k,True,1)
    r2score_old = 0

    #B_Best = np.zeros((deg_max,int((deg_max+1)*(deg_max+2)/2)))

    r2_scores = np.zeros((deg_max,k))
    r2_score = np.zeros(deg_max)
    MSEs = np.zeros((deg_max,k))
    MSE = np.zeros(deg_max)

    i = 0
    for deg in range(1,deg_max+1):
        j = 0
        for tr_idx, ts_idx in kfold.split(x):
            x_train = x[tr_idx]
            y_train = y[tr_idx]
            z_train = z[tr_idx]

            x_test = x[ts_idx]
            y_test = y[ts_idx]
            z_test = z[ts_idx]

            X_Train = X_DesignMatrix(x_train,y_train,deg)
            X_Test = X_DesignMatrix(x_test,y_test,deg)
            B = Coeff(X_Train,z_train)
            #print(B)

            z_predict_train = X_Train @ B
            z_predict_test = X_Test @ B
            z_true = FrankeFunction(x_test,y_test,len(x_test),0,False)

            r2_scores[i,j] = metrics.r2_score(z_true, z_predict_test)
            MSEs[i,j] = metrics.mean_squared_error(z_true, z_predict_test)
            j += 1
        r2_score[i] = np.mean(r2_scores[i,:])
        MSE[i] = np.mean(MSEs[i,:])
        i += 1

    return r2_score, MSE


def main():

    np.random.seed(0)


    n = 200
    noise = 1
    test_S = 0.3
    degree = 5
    k = 5
    deg_max = 20
    lmb = 0.000001

    x = np.sort(np.random.uniform(0,1,n))
    y = np.sort(np.random.uniform(0,1,n))
    x,y = np.meshgrid(x,y)
    x_1 = np.ravel(x)
    y_1 = np.ravel(y)

    z = FrankeFunction(x,y,n,noise)
    z_1 = np.ravel(z)
    z_true = FrankeFunction(x,y,n,noise,False)

    z_2 = DataImport('Norway_1arc.tif')

    X_DM = X_DesignMatrix(x_1,y_1,degree)

    B = Coeff(X_DM,z_1)
    z_predict = np.dot(X_DM,B).reshape(n,n)
    r2_score = metrics.r2_score(z_true,np.reshape(z_predict,(n,n)))
    MSE = metrics.mean_squared_error(z_true,np.reshape(z_predict,(n,n)))
    print("MSE = %s, R2 score = %s" %(MSE,r2_score))

    B_ridge = CoeffRidge(X_DM,z_1,lmb)
    z_ridge = np.dot(X_DM,B_ridge).reshape(n,n)
    r2_score_ridge = metrics.r2_score(z_true,np.reshape(z_ridge,(n,n)))
    MSE_ridge = metrics.mean_squared_error(z_true,np.reshape(z_ridge,(n,n)))
    print("MSE ridge = %s, R2 score ridge = %s" %(MSE_ridge,r2_score_ridge))

    clf_lasso = skl.Lasso(alpha=lmb).fit(X_DM,z_1)
    z_lasso = clf_lasso.predict(X_DM)
    #z_lasso = X_DM @ B_lasso
    r2_score_lasso = metrics.r2_score(z_true,np.reshape(z_lasso,(n,n)))
    MSE_lasso = metrics.mean_squared_error(z_true,np.reshape(z_lasso,(n,n)))
    print("MSE lasso = %s, R2 score lasso = %s" %(MSE_lasso,r2_score_lasso))


    plotting_function(x,y,z,n)
    plotting_function(x,y,z_predict,n)
    plotting_function(x,y,z_ridge,n)
    plotting_function(x,y,z_lasso,n)

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x_1,y_1,z_1,test_size=test_S)
    r2_score, MSE = kCrossValidation(x_train,y_train,z_train,deg_max,k)

    degs = range(1,deg_max+1)

    plt.figure()
    #plt.plot(degs, r2_score, degs, MSE)
    plt.semilogy(degs, r2_score, degs, MSE)

if __name__ == "__main__":
    main()
    plt.show()
