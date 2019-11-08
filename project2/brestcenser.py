import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
#from NN import *
from p2NN import *
from sklearn.metrics import accuracy_score
import seaborn as sns


def Gradient_Decent(X, Y, eta = 0.01, epochs = 20, batches = 100, max_iter = 1000, min_error = 1e-7):
    Beta = np.random.randn(len(X[0]))#np.random.randn(len(X[:,0]))
    Beta_old = Beta
    convergence = 1

    """
    def learningrates(t):
        t0 = 5
        t1 = 50
        return t0/(t+t1)
    """
    print(np.shape(X_train),np.shape(Y_train))
    def classification(X,Y,Beta):
        p = 1/(1+np.exp(-Xc@Beta))
        gradient = -(np.dot(Xc.T,Yc-p))
        return gradient

    for i in range(epochs):
        j = 0
        while j < max_iter and convergence != 0:
            chosen_points = np.random.choice(np.shape(X[0]),size=batches)#np.shape(X[0]))#len(Beta))
            Xc = X[chosen_points]#:chosen_points+1]
            Yc = Y[chosen_points]#:chosen_points+1]

            #print(np.shape(Xc), np.shape(Yc), np.shape(Beta))

            gradient = classification(Xc,Yc,Beta)
            #eta = learningrates(i*len(X[:,0]+j))
            Beta = Beta - eta*gradient

            j += 1
        """
        if crossentropy(Beta) < crossentropy(Beta_old):
            Beta_old = Beta
        """

    probabilities = np.exp(X@Beta)/(1+np.exp(X@Beta))
    Y_predict = (probabilities >= 0.5).astype(int)
    return Y_predict

def accuracy(Y,Y_tilde):
    return np.mean( Y == Y_tilde)

np.random.seed(42)

X,Y = datasets.make_classification(1000)


print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7,
                                                    test_size=0.3)

prediction1 = Gradient_Decent(X_train,Y_train,0.001,max_iter=1000,batches=700)
print(prediction1)
print(accuracy(Y_train,prediction1))
eta_vals = np.logspace(-5, 1, 7)
print(np.shape(X_train),np.shape(Y_train))
"""
for eta in eta_vals:
    prediction1 = Gradient_Decent(X_train,Y_train,eta)
    #print(prediction1)
"""

alpha = 0.0001
epochs = 5
batch_size = 50
hidden_neurons = 100
categories = 1
max_iter = 300
error_min = 0.01
lmbd_vals = np.logspace(-5, 1, 7)
# store the models for later use
DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)

#print(np.shape(X),np.shape(Y))
#print(np.shape(X_train),np.shape(Y_train))

# grid search
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):

        dnn = NeuralNetwork(X_data=X_train, Y_data=Y_train,
        hidden_neurons=hidden_neurons, categories=categories,
        epochs=epochs, batches=batch_size, max_iter=max_iter,
        error_min=error_min, eta=eta, lmbd=lmbd, alpha=alpha)
        dnn.SGD()
        test_predict = dnn.predict(X_test)

        DNN_numpy[i][j] = dnn


        print("Learning rate  = ", eta)
        print("Lambda = ", lmbd)
        print("Accuracy score on test set: ", accuracy_score(Y_test, test_predict))
        print()


sns.set()

train_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))
test_accuracy = np.zeros((len(eta_vals), len(lmbd_vals)))

for i in range(len(eta_vals)):
    for j in range(len(lmbd_vals)):
        dnn = DNN_numpy[i][j]

        train_pred = dnn.predict(X_train)
        test_pred = dnn.predict(X_test)

        train_accuracy[i][j] = accuracy_score(Y_train, train_pred)
        test_accuracy[i][j] = accuracy_score(Y_test, test_pred)

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
