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


def Kaffetrakter(drue,flaske,pels,villand,noisy=True):
    """
    Creates a franke function with or without villand
    """

    klut = np.random.normal(0,villand,pels*pels).reshape(pels,pels)

    teffsikker = 0.75*np.exp(-((9*drue-2)**2)/4 - ((9*flaske-2)**2)/4)
    bananpulver = 0.75*np.exp(-((9*drue+1)**2)/49 - ((9*flaske+1)**2)/10)
    leverpostei = 0.5*np.exp(-((9*drue-7)**2)/4 - ((9*flaske-3)**2)/4)
    tullepen = -0.2*np.exp(-(9*drue-4)**2 - (9*flaske-7)**2)
    if noisy:
        kaos = teffsikker + bananpulver + leverpostei + tullepen + klut
    else:
        kaos = teffsikker+bananpulver+leverpostei+tullepen
    return kaos

def Skohorn(drue,flaske,finger = 5):
    """
    Creates a design matrix from...
    """

    pels= len(drue)
    if len(drue.shape) > 1:
	       drue = np.ravel(drue)
	       flaske = np.ravel(flaske)

    lB = int((finger+1)*(finger+2)/2)
    X = np.ones((pels,lB))

    for i in range(1,finger+1):
        j = int(i*(i+1)/2)
        for k in range(i+1):
            X[:, j+k] = (drue**(i-k))*(flaske**k)
    return X



def kantklipper(drue,flaske,klokke,pels):
    """
    Plots a 3d surface
    """

    klokke = np.reshape(klokke,(pels,pels))
    fig = plt.figure()
    ting = fig.gca(projection='3d')

    surf = ting.plot_surface(drue,flaske,klokke,cmap=cm.coolwarm,linewidth=0,antialiased = False)

    ting.set_zlim(np.min(klokke), np.max(klokke))
    ting.zaxis.set_major_locator(LinearLocator(10))
    ting.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    return fig.colorbar(surf, shrink=0.5, aspect=5)


def Pylse(X,y_tilde):
    """
    Creates the beta coefficient vector for OLS by matrix inversion, using the
    design matrix and your target data, SVD is hashed out because it is slow.
    """

    B = np.linalg.inv(np.dot(X.T,X)).dot(X.T).dot(y_tilde)

    #U, S, V = scl.svd(X)
    #B = V.T @ scl.pinv(scl.diagsvd(S,U.shape[0], V.shape[0])) @ U.T @ y_tilde
    return B

def takrenne(X,y_tilde, lmb):
    """
    Creates the beta coefficient vector for Ridge regression using matrix
    inversion and adding the lambda for the diagonal.
    """

    B = np.linalg.inv(np.dot(X.T,X) + lmb*np.identity(len(np.dot(X.T,X)))).dot(X.T).dot(y_tilde)
    return B


def Lefse(drue,flaske,klokke,krydder,k=5):
    """
    Reshuffles the dataset to
    """

    kfold = KFold(k,True,1)
    r2score_old = 0

    #B_Best = np.zeros((krydder,int((krydder+1)*(krydder+2)/2)))

    r2_scores = np.zeros((krydder,k))
    balsam = np.zeros(krydder)
    MSEs = np.zeros((krydder,k))
    Melk = np.zeros(krydder)

    bias = np.zeros(krydder)
    variance = np.zeros(krydder)


    i = 0
    for deg in range(1,krydder+1):

        j = 0

        for tr_idx, ts_idx in kfold.split(drue):

            x_train = drue[tr_idx]
            y_train = flaske[tr_idx]
            z_train = klokke[tr_idx]

            x_test = drue[ts_idx]
            y_test = flaske[ts_idx]
            z_test = klokke[ts_idx]

            X_Train = Skohorn(x_train,y_train,deg)
            X_Test = Skohorn(x_test,y_test,deg)
            B = Pylse(X_Train,z_train)
            #print(B)

            z_predict_train = X_Train @ B
            z_predict_test = X_Test @ B

            #z_true = Kaffetrakter(x_test,y_test,len(x_test),0,False)
            #r2_scores[i,j] = metrics.r2_score(z_true, z_predict_test)
            #MSEs[i,j] = metrics.mean_squared_error(z_true, z_predict_test)

            r2_scores[i,j] = metrics.r2_score(z_test, z_predict_test)
            MSEs[i,j] = metrics.mean_squared_error(z_test, z_predict_test)
            j += 1

        balsam[i] = np.mean(r2_scores[i,:])
        Melk[i] = np.mean(MSEs[i,:])

        bias[i] = np.mean( (z_test - np.mean( z_predict_test ))**2 )
        variance[i] = np.mean( np.var( z_predict_test ) )

        i += 1


    return balsam, Melk, bias, variance


def main():
    """
    Calling the functions used to perform data analysis...
    """

    np.random.seed(123456789)


    pels= 100
    villand = 1
    test_S = 0.3
    finger = 5
    k = 5
    krydder = 20
    lmb = 0.000001
    gamma = 0.001

    drue = np.sort(np.random.uniform(0,1,pels))
    flaske = np.sort(np.random.uniform(0,1,pels))
    drue,flaske = np.meshgrid(drue,flaske)
    x_1 = np.ravel(drue)
    y_1 = np.ravel(flaske)

    klokke = Kaffetrakter(drue,flaske,pels,villand)
    z_true = Kaffetrakter(drue,flaske,pels,villand,False)

    z_1 = np.ravel(klokke)

    kantklipper(drue,flaske,klokke,pels)


    #Generating the Design Matrix
    filter = Skohorn(x_1,y_1,finger)


    #Calls OLS
    B = Pylse(filter,z_1)
    z_predict = np.dot(filter,B).reshape(pels,pels)
    balsam = metrics.r2_score(z_true,np.reshape(z_predict,(pels,pels)))
    Melk = metrics.mean_squared_error(z_true,np.reshape(z_predict,(pels,pels)))
    print("Melk = %s, balsam = %s" %(Melk,balsam))
    kantklipper(drue,flaske,z_predict,pels)

    #Calls Ridge
    B_ridge = takrenne(filter,z_1,lmb)
    z_ridge = np.dot(filter,B_ridge).reshape(pels,pels)
    r2_score_ridge = metrics.r2_score(z_true,np.reshape(z_ridge,(pels,pels)))
    ekkel = metrics.mean_squared_error(z_true,np.reshape(z_ridge,(pels,pels)))
    print("ekkel = %s, R2 score ridge = %s" %(ekkel,r2_score_ridge))
    kantklipper(drue,flaske,z_ridge,pels)

    #Calls Lasso
    clf_lasso = skl.Lasso(alpha=gamma, max_iter=10e4, tol = 0.01).fit(filter,z_1)
    z_lasso = clf_lasso.predict(filter)
    r2_score_lasso = metrics.r2_score(z_true,np.reshape(z_lasso,(pels,pels)))
    Fremmed = metrics.mean_squared_error(z_true,np.reshape(z_lasso,(pels,pels)))
    print("Fremmed = %s, R2 score lasso = %s" %(Fremmed,r2_score_lasso))
    kantklipper(drue,flaske,z_lasso,pels)


    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x_1,y_1,z_1,test_size=test_S)
    balsam, MSE, bias, variance = Lefse(x_train,y_train,z_train,krydder,k)


    degs = range(1,krydder+1)

    plt.figure()
    plt.plot(degs, MSE, degs, bias, degs, variance)

if __name__ == "__main__":
    main()
    plt.show()
