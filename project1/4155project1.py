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
import sys

def FrankeFunction(x,y,n,noise,noisy=True):
    """
    Creates a franke function with or without noise.
    """

    epsilon = np.random.normal(0,noise,(n,n))

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
    """
    Imports and downscales the terraindata, then plots it.
    """
    sc = 10
    # Load the terrain
    terrain1 = imread(filename)

    # Downscale the terrain
    downscaled = terrain1[1::sc,1::sc]


    #Show the downscaled terrain
    #plt.figure()
    #plt.imshow(terrain1, cmap='gray')
    plt.figure()
    plt.imshow(downscaled, cmap='gray')

    return downscaled


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


def X_DesignMatrix(x,y,degree = 5):
    """
    Creates a design matrix of polynomials from parameters x and y (Vandermonde).
    """

    if len(x.shape) > 1:
	       x = np.ravel(x)
	       y = np.ravel(y)

    n = len(x)
    lB = int((degree+1)*(degree+2)/2)
    X = np.ones((n,lB))

    for i in range(1,degree+1):
        j = int(i*(i+1)/2)
        for k in range(i+1):
            X[:, j+k] = (x**(i-k))*(y**k)
    return X


def Coeff(X,y_tilde):
    """
    Creates the beta coefficient vector for OLS by matrix inversion, using the
    design matrix and your target data.
    """

    B = np.linalg.pinv(np.dot(X.T,X)).dot(X.T).dot(y_tilde)

    #U, S, V = scl.svd(X)
    #B = V.T @ scl.pinv(scl.diagsvd(S,U.shape[0], V.shape[0])) @ U.T @ y_tilde
    return B


def CoeffRidge(X,y_tilde, lmb):
    """
    Creates the beta coefficient vector for Ridge regression using matrix
    inversion and adding the lambda for the diagonal.
    """

    B = np.linalg.pinv(np.dot(X.T,X) + lmb*np.identity(len(np.dot(X.T,X)))).dot(X.T).dot(y_tilde)
    return B

def ConfInterval(B, X, c=1.96):
    """
    Finds the confidence interval of the Betas.
    """

    SE = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)))*c
    B_min = B - SE
    B_max = B + SE
    return B_min, B_max

def kCrossValidation(x,y,z,deg_max,k=5,lmb=0):
    """
    Performs a k-fold crossvalidation of a dataset.
    """

    kfold = KFold(k,True,1)
    r2score_old = 0

    r2_scores = np.zeros((deg_max+1,k))
    r2_score = np.zeros(deg_max+1)
    MSEs_test = np.zeros((deg_max+1,k))
    MSE_test = np.zeros(deg_max+1)
    MSEs_train = np.zeros((deg_max+1,k))
    MSE_train = np.zeros(deg_max+1)

    bias = np.zeros(deg_max+1)
    variance = np.zeros(deg_max+1)

    i = 0
    for deg in range(0,deg_max+1):

        j = 0
        b = 0
        v = 0

        for tr_idx, ts_idx in kfold.split(x):

            x_train = x[tr_idx]
            y_train = y[tr_idx]
            z_train = z[tr_idx] #np.ravel(z[tr_idx])

            x_test = x[ts_idx]
            y_test = y[ts_idx]
            z_test = z[ts_idx] #np.ravel(z[ts_idx])

            X_Train = X_DesignMatrix(x_train,y_train,deg)
            X_Test = X_DesignMatrix(x_test,y_test,deg)
            B = CoeffRidge(X_Train,z_train,lmb)
            #print(B)

            z_predict_train = X_Train @ B
            z_predict_test = X_Test @ B

            #z_true = FrankeFunction(x_test,y_test,len(x_test),0,False)
            #r2_scores[i,j] = metrics.r2_score(z_true, z_predict_test)
            #MSEs_test[i,j] = metrics.mean_squared_error(z_true, z_predict_test)

            r2_scores[i,j] = metrics.r2_score(z_test, z_predict_test)
            MSEs_test[i,j] = metrics.mean_squared_error(z_test, z_predict_test)
            MSEs_train[i,j] = metrics.mean_squared_error(z_train, z_predict_train)
            b += (z_test - np.mean( z_predict_test ))**2
            v += np.var( z_predict_test )

            j += 1

        r2_score[i] = np.mean(r2_scores[i,:])
        MSE_test[i] = np.mean(MSEs_test[i,:])
        MSE_train[i] = np.mean(MSEs_train[i,:])

        bias[i] = np.mean( b )/k
        variance[i] = np.mean( v )/k

        i += 1


    return r2_score, MSE_test, MSE_train, bias, variance


def main():
    """
    Performs Data analysis on the Franke function with noise, using OLS, Ridge
    and Lasso with crossvalidation.
    """

    np.random.seed(42)


    n = 100
    noise = 0.5
    test_S = 0.3
    degree = 5
    k = 5
    deg_max = 12 #20
    lmb = 0.001
    gamma = 0.0001

    lmb_range = lmb*np.ones(13)
    for i in range(1, len(lmb_range)): lmb_range[i] = np.sqrt(10)*lmb_range[i-1]

    x = np.sort(np.random.uniform(0,1,n))
    y = np.sort(np.random.uniform(0,1,n))
    x,y = np.meshgrid(x,y)
    x_1 = np.ravel(x)
    y_1 = np.ravel(y)

    z = FrankeFunction(x,y,n,noise)
    z_true = FrankeFunction(x,y,n,noise,False)

    z_1 = np.ravel(z)

    plotting_function(x,y,z,n)


    #Generating the Design Matrix
    X_DM = X_DesignMatrix(x_1,y_1,degree)


    #Calls for solving tasks with OLS
    B = Coeff(X_DM,z_1)
    z_predict = np.dot(X_DM,B).reshape(n,n)
    r2_score = metrics.r2_score(z_true,np.reshape(z_predict,(n,n)))
    MSE = metrics.mean_squared_error(z_true,np.reshape(z_predict,(n,n)))
    #r2_score = metrics.r2_score(z,np.reshape(z_predict,(n,n)))
    #MSE = metrics.mean_squared_error(z,np.reshape(z_predict,(n,n)))
    print("MSE = %s, R2 score = %s" %(MSE,r2_score))
    plotting_function(x,y,z_predict,n)
    B_max, B_min = ConfInterval(B,X_DM)
    plt.figure()
    plt.plot(range(len(B)), B, label="$\beta$")
    plt.plot(range(len(B)), B_max, label="$\beta_min$")
    plt.plot(range(len(B)), B_min, label="$\beta_min$")

    #Calls for solving tasks with Ridge
    B_ridge = CoeffRidge(X_DM,z_1,lmb)
    z_ridge = np.dot(X_DM,B_ridge).reshape(n,n)
    r2_score_ridge = metrics.r2_score(z_true,np.reshape(z_ridge,(n,n)))
    MSE_ridge = metrics.mean_squared_error(z_true,np.reshape(z_ridge,(n,n)))
    #r2_score_ridge = metrics.r2_score(z,np.reshape(z_ridge,(n,n)))
    #MSE_ridge = metrics.mean_squared_error(z,np.reshape(z_ridge,(n,n)))
    print("MSE ridge = %s, R2 score ridge = %s" %(MSE_ridge,r2_score_ridge))
    plotting_function(x,y,z_ridge,n)

    #Calls for solving tasks with Lasso
    clf_lasso = skl.Lasso(alpha=gamma, max_iter=10e4, tol = 0.01).fit(X_DM,z_1)
    z_lasso = clf_lasso.predict(X_DM)
    r2_score_lasso = metrics.r2_score(z_true,np.reshape(z_lasso,(n,n)))
    MSE_lasso = metrics.mean_squared_error(z_true,np.reshape(z_lasso,(n,n)))
    print("MSE lasso = %s, R2 score lasso = %s" %(MSE_lasso,r2_score_lasso))
    plotting_function(x,y,z_lasso,n)


    #x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x_1,y_1,z_1,test_size=test_S)
    #r2_score, MSE_test, MSE_train, bias, variance = kCrossValidation(x_train,y_train,z_train,deg_max,k)
    r2_score, MSE_test, MSE_train, bias, variance = kCrossValidation(x_1,y_1,z_1,deg_max,k)

    degs = range(0,deg_max+1)

    plt.figure()
    plt.plot(degs, (MSE_test), label="test MSE")
    plt.plot(degs, (MSE_train), label="train MSE")
    plt.plot(degs, (bias), label="Bias")
    plt.plot(degs, (variance), label="Variance")
    plt.legend()


    i = 0
    MSE_test_ridge = np.zeros((len(lmb_range),deg_max+1))
    MSE_train_ridge = np.zeros((len(lmb_range),deg_max+1))


    print("Crossvalidation with ridge completion:")
    for l in lmb_range:
        sys.stdout.write("\r%d %%" %(100*(i+1)/len(lmb_range)))
        r2_score_ridge, MSE_test_ridge[i,:], MSE_train_ridge[i,:], bias_ridge, variance_ridge = kCrossValidation(x_1,y_1,z_1,deg_max,k,l)
        i+=1
        sys.stdout.flush()


    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    #ax.add_collection3d(lmb_range,MSE_test_ridge)
    msx,msy = np.meshgrid(lmb_range,degs)
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(np.log10(msx),msy,(MSE_test_ridge),label="Test MSE",linewidth=0,antialiased = False)
    surf = ax.plot_surface(np.log10(msx),msy,(MSE_train_ridge),label="Train MSE",linewidth=0,antialiased = False)

    ax.set_zlim(np.min((MSE_test_ridge)), np.max((MSE_test_ridge)))
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    plt.title("Crossvalidation of Ridge")

    fig.colorbar(surf, shrink=0.5, aspect=5)

    #print(MSE_test_ridge)


def main2():
    """
    Solves the same tasks as above but for the real terrain data imported with
    the function DataImport.
    """

    TerrainDataSet = DataImport('Norway_1arc.tif')
    ny = len(TerrainDataSet[:,0])
    nx = len(TerrainDataSet[0,:])


    Tdegree = 50
    Tlambda = 1e-4
    Tgamma = 5e-6

    PosX = np.linspace(0,1,nx)
    PosY = np.linspace(0,1,ny)
    PosX,PosY = np.meshgrid(PosX,PosY)

    PosZ = np.ravel(TerrainDataSet)/np.max(TerrainDataSet)
    X_Terrain = X_DesignMatrix(PosX,PosY,Tdegree)


    B_OLS_Terrain = Coeff(X_Terrain, PosZ)
    zp_OLS_Terrain = np.dot(X_Terrain,B_OLS_Terrain).reshape(ny,nx)

    plt.figure()
    plt.imshow( zp_OLS_Terrain , cmap="gray")


    B_Ridge_Terrain = CoeffRidge(X_Terrain, PosZ, Tlambda)
    zp_Ridge_Terrain = np.dot(X_Terrain,B_Ridge_Terrain).reshape(ny,nx)

    plt.figure()
    plt.imshow( zp_Ridge_Terrain , cmap="gray")
    #print(zp_Ridge_Terrain)


    B_Lasso_Terrain = skl.Lasso(alpha=Tgamma, max_iter=10e4, tol = 0.01).fit(X_Terrain, PosZ)
    # zp_Lasso = clf_lasso.predict(X_Terrain)
    zp_Lasso = B_Lasso_Terrain.predict(X_Terrain)
    zp_Lasso_Terrain = zp_Lasso.reshape(ny,nx)

    plt.figure()
    plt.imshow( zp_Lasso_Terrain , cmap='gray')
    print(zp_Lasso_Terrain)



if __name__ == "__main__":
    #Recommended not running both mains at the same time.
    main()
    ##main2()
    plt.show()
