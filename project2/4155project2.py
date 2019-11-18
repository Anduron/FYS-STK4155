import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sklearn.linear_model as skl
import sklearn.neural_network as skn
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, roc_auc_score
import seaborn as sns
import pandas as pd
import sys, os
import tensorflow as tf
from NN import *


def Gradient_Decent(X, Y, eta = 0.01, epochs = 20, batches = 100, max_iter = 1000, min_error = 1e-7):
    Beta = np.random.randn(len(X[0]),1)#np.random.randn(len(X[:,0]))
    convergence = 1

    def learningrates(t):
        t0 = 5
        t1 = 50
        return t0/(t+t1)

    #print(np.shape(X_train),np.shape(Y_train))
    def classification(X,Y,Beta):
        sig = 1/(1+np.exp(-Xc@Beta))
        gradient = -(np.dot(Xc.T,Yc-sig))
        return gradient

    for i in range(epochs):
        j = 0
        while j < max_iter and convergence != 0:
            chosen_points = np.random.choice(len(Beta))
            Xc = X[chosen_points:chosen_points+1]
            Yc = Y[chosen_points:chosen_points+1]

            #print(np.shape(Xc), np.shape(Yc), np.shape(Beta))

            gradient = classification(Xc,Yc,Beta)
            eta = learningrates(i*len(X[:,0]+j))
            Beta = Beta - eta*gradient

            j += 1

    probabilities = np.exp(X@Beta)/(1+np.exp(X@Beta))
    Y_predict = (probabilities >= 0.5).astype(int)
    return Y_predict

def sigmoid(x):
    return 1/(1+np.exp(-x))

def DesignMatrix(X):
    X = np.c_[np.ones(len(X[:,0])),X]
    return X

def accuracy(Y,Y_tilde):
    return np.mean( Y == Y_tilde)


def main():

    np.random.seed(42)

    dir = os.getcwd()
    filename = dir + "/default of credit card clients.xls"
    nanDict = {}
    df = pd.read_excel(filename, header = 1, skiprows = 0, index_col = 0, na_values = nanDict)
    scale = StandardScaler()

    df.rename(index=str, columns={"default payment next month": "defaultPaymentNextMonth"}, inplace = True)

    y_check = df.loc[:, df.columns == "defaultPaymentNextMonth"].values
    i = 0
    for j in range(len(y_check)):
        if y_check[j] == 0:
            i+=1
    print(i/len(y_check))

    df = df.drop(df[(df.BILL_AMT1 == 0)&
                    (df.BILL_AMT2 == 0)&
                    (df.BILL_AMT3 == 0)&
                    (df.BILL_AMT4 == 0)&
                    (df.BILL_AMT5 == 0)&
                    (df.BILL_AMT6 == 0)].index)

    df = df.drop(df[(df.PAY_AMT1 == 0)&
                    (df.PAY_AMT2 == 0)&
                    (df.PAY_AMT3 == 0)&
                    (df.PAY_AMT4 == 0)&
                    (df.PAY_AMT5 == 0)&
                    (df.PAY_AMT6 == 0)].index)

    y_check = df.loc[:, df.columns == "defaultPaymentNextMonth"].values
    i = 0
    for j in range(len(y_check)):
        if y_check[j] == 0:
            i+=1
    print(i/len(y_check))

    X = df.loc[:, df.columns != "defaultPaymentNextMonth"].values
    y = df.loc[:, df.columns == "defaultPaymentNextMonth"].values

    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap="viridis")
    plt.show()

    A = OneHotEncoder(categories="auto", sparse=False)

    X = ColumnTransformer([("",A,[1,2,3,5,6,7,8,9,10]),], remainder = "passthrough").fit_transform(X)

    X_Train, X_Test, y_Train, y_Test = train_test_split(X,y,train_size=0.8, test_size=0.2)

    X_Train = scale.fit_transform(X_Train)
    X_Test = scale.transform(X_Test)

    print(np.shape(y_Train))
    Y_train_onehot, Y_test_onehot = A.fit_transform(y_Train), A.fit_transform(y_Test)

    XDM_Train = DesignMatrix(X_Train)
    Y_predict = Gradient_Decent(XDM_Train, Y_train_onehot,eta=0.01,max_iter=10000)
    print(Y_predict)

    all_pay_init = np.ones((len(y_Test),2))
    all_pay = all_pay_init@np.array([[1,0],[0,0]])
    if all_pay[:,0].all() == 1 and all_pay[:,1].all() == 0:
        print("yes")
    print(all_pay)
    all_pay2 = np.zeros(len(y_Test))
    print(accuracy_score(y_Test,all_pay2))
    print(accuracy(Y_train_onehot,Y_predict))


    params = [{'C': 1/(np.logspace(-2,5)), 'solver':['lbfgs']}]
    logr = skl.LogisticRegression()
    logr.fit(X_Train,y_Train)
    prediction2 = logr.predict(X_Test)
    print(logr.score(X_Test,y_Test))

    print(prediction2)

    reg = skn.MLPClassifier(hidden_layer_sizes=(100,10),learning_rate="adaptive",learning_rate_init = 0.001,max_iter=1000,tol=1e-7,verbose=False)
    reg = reg.fit(X_Train,Y_train_onehot)
    pred = reg.predict(X_Test)
    scr = reg.score(X_Test,Y_test_onehot)
    print(scr)

    epochs = 40
    batch_size = 100
    categories = 10
    max_iter = 200
    error_min = 0.01
    eta_vals = np.logspace(-5, 0, 6)
    lmbd_vals = np.logspace(-5, 1, 7)

    # store the models for later use
    DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
    #print(len(X_Train[0]))

    layers = [len(X_Train[0]), 48, 12, len(Y_train_onehot[0])] #change to 16
    activations = ["sigmoid","ReLu","softmax"]#,"sigmoid","sigmoid","softmax"]
    #print(len(activations))

    # grid search
    for i, eta in enumerate(eta_vals):
        for j, lmbd in enumerate(lmbd_vals):
            alpha = 0.01
            dnn = NeuralNetwork(X_Train,Y_train_onehot,layers, activations,
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
            test_predict = dnn.predict(X_Test)
            print(test_predict)
            DNN_numpy[i][j] = dnn

            print("Learning rate  = ", eta)
            print("Lambda = ", lmbd)
            #print("Accuracy score on test set: ", accuracy_score(y_Test, test_predict))
            print("AUC score on test set: ", roc_auc_score(y_Test, test_predict))
            print()


    sns.set()

    train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
    test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

    for i in range(len(eta_vals)):
        for j in range(len(lmbd_vals)):
            dnn = DNN_numpy[i][j]

            train_pred = dnn.predict(X_Train)
            test_pred = dnn.predict(X_Test)

            train_accuracy[i][j] = roc_auc_score(y_Train, train_pred)
            test_accuracy[i][j] = roc_auc_score(y_Test, test_pred)


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

if __name__== "__main__":
    main()
