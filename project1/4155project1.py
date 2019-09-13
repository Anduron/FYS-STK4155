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

    ax.set_zlim(-0.10, 1.40)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    return fig.colorbar(surf, shrink=0.5, aspect=5)


def Coeff(X,y_tilde):
    B = np.linalg.inv(np.dot(X.T,X)).dot(X.T).dot(y_tilde)
    return B


def kCrossValidation(x,y,z,deg_max,k=5):

    kfold = KFold(k,True,1)
    r2score_old = 0
    i = 0
    B_Best = np.zeros((deg_max,int((deg_max+1)*(deg_max+2)/2)))

    r2_scores = np.zeros(deg_max)
    MSEs = np.zeros(deg_max)

    for deg in range(1,deg_max+1):
        j = 0
        for tr_idx, ts_idx in kfold.split(x):
            x_train = x[tr_idx]
            y_train = y[tr_idx]
            z_train = np.ravel(z[tr_idx])

            x_test = x[ts_idx]
            y_test = y[ts_idx]
            z_test = np.ravel(z[ts_idx])

            X_Train = X_DesignMatrix(x_train,y_train,deg)
            X_Test = X_DesignMatrix(x_test,y_test,deg)
            B = Coeff(X_Train,z_train)

            z_predict_train = X_Train @ B
            z_predict_test = X_Test @ B


            r2score = metrics.r2_score(z_test,z_predict_test)
            MSE = metrics.mean_squared_error(z_test,z_predict_test)

            if r2score > r2score_old:
                B_Best = Coeff(X_Train,z_train)
                r2_scores[i] = r2score
                MSEs[i] = MSE
            j += 1
        i += 1
        r2score_old = r2score
    print(MSEs)
    return B_Best, r2_scores, MSEs

def main():

    np.random.seed(0)


    n = 100
    noise = 0.01
    test_S = 0.3
    degree = 5
    k = 5
    deg_max = 20

    x = np.sort(np.random.uniform(0,1,n))
    y = np.sort(np.random.uniform(0,1,n))
    x,y = np.meshgrid(x,y)
    x_1 = np.ravel(x)
    y_1 = np.ravel(y)

    z = FrankeFunction(x,y,n,noise)
    z_1 = np.ravel(z)

    X_DM = X_DesignMatrix(x_1,y_1,degree)

    #z_predict, varB = OLS(X_DM,z_1)
    #r2_score = metrics.r2_score(z,np.reshape(z_predict,(n,n)))
    #MSE = metrics.mean_squared_error(z,np.reshape(z_predict,(n,n)))
    #print("MSE = %s, R2 score = %s, Variance B = %s" %(MSE,r2_score,varB))

    #plotting_function(x,y,z,n)
    #plotting_function(x,y,z_predict,n)

    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(x_1,y_1,z_1,test_size=test_S)
    B_Best, r2_scores, MSE = kCrossValidation(x_train,y_train,z_train,deg_max,k)

    degs = range(1,deg_max+1)
    #r2_scores = np.zeros(len(degs))
    #for i in degs:
    #    X_Test = X_DesignMatrix(x_test,y_test,i)
    #    z_predict = X_Test @ B_Best[i,:]
    #    r2_scores[i] = metrics.r2_score(z_test,z_predict)

    plt.plot(degs, r2_scores, degs, MSE)

if __name__ == "__main__":
    main()
    plt.show()
