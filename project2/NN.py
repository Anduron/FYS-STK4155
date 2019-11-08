import numpy as np

class NeuralNetwork:
    def __init__(self,
        X,
        y,
        layers,
        activations,
        epochs=10,
        batches=100,
        max_iter = 100,
        error_min=1e-6,
        eta=0.1,
        lmbd=0.0,
        alpha=1.0):

        self.layers = [layers]
        self.activations = activations
        self.X_data = X
        self.y_data = y

        self.epochs = epochs
        self.batches = batches
        self.max_iter = max_iter
        self.error_min = error_min

        self.eta = eta
        self.lmbd = lmbd
        self.alpha = alpha

        self.biases = [0.01*np.ones(y) for y in layers[1:]]
        self.weights = [np.random.randn(x,y) for x, y in zip(layers[:-1],layers[1:])]

    def activation(self, activations, z):
        if activations == "tanh":
            return np.tanh(z)
        elif activations == "sigmoid":
            return 1/(1+np.exp(-z))
        elif activations == "Leaky ReLu":
            return np.maximum(self.alpha*z,z)
        elif activations == "ReLu":
            return np.maximum(0,z)
        elif activations == "ELU":
            return np.where(z<0,self.alpha*(np.exp(z)-1),z)
        elif activations == "linear":
            return z
        elif activations == "softmax":
            return np.exp(z)/np.sum(np.exp(z), axis=1, keepdims=True)

    def activation_derivative(self, activations, a):
        if activations == "tanh":
            return 1-a**2
        elif activations == "sigmoid":
            return a*(1-a)
        elif activations == "Leaky ReLu":
            return np.where(a>0,1,self.alpha)
        elif activations == "ReLu":
            return np.where(a>0,1,0)
        elif activations == "ELU":
            return np.where(a>0,1,self.alpha*(a+1))
        elif activations == "linear":
            return 1
        elif activations == "softmax":
            return 1

    def FeedForward(self):
        # feed-forward for training
        self.z = [np.zeros(i.shape) for i in self.biases]
        self.a = [np.zeros(j.shape) for j in self.biases]

        self.z[0] = np.dot(self.X,self.weights[0])+self.biases[0]
        self.a[0] = self.activation(self.activations[0],self.z[0])

        for i in range(1,len(self.biases)):
            self.z[i] = np.dot(self.a[i-1],self.weights[i]) + self.biases[i]
            self.a[i] = self.activation(self.activations[i],self.z[i])
        self.a_o = self.a[-1]

        #if self.a_o.all() == (np.exp(self.z[-1])/np.sum(np.exp(self.z[-1]),axis=1,keepdims=True)).all():
        #    print(1)

    def BackwardPropogation(self):
        self.biases_gradient = [np.zeros(i.shape) for i in self.biases]
        self.weights_gradient = [np.zeros(j.shape) for j in self.weights]

        error = [np.zeros(i.shape) for i in self.biases]
        error_output = (self.a_o - self.y)*self.activation_derivative(self.activations[-1],self.a[-1])
        error[-1] = error_output

        for i in range(len(self.biases)-1,0,-1):
            error[i-1] = np.dot(error[i],self.weights[i].T)*self.activation_derivative(self.activations[i-1],self.a[i-1])
            self.biases_gradient[i] = np.sum(error[i], axis=0)
            self.weights_gradient[i] = np.dot(self.a[i-1].T,error[i])

            if self.lmbd > 0.0:
                self.weights_gradient[i] += self.lmbd*self.weights[i]

            #print(np.shape(self.weights_gradient[i]))
            self.weights[i] -= self.eta*self.weights_gradient[i]
            self.biases[i] -= self.eta*self.biases_gradient[i]

        self.weights_gradient[0] = np.dot(self.X.T, error[0])
        self.biases_gradient[0] = np.sum(error[0], axis=0)

        if self.lmbd > 0.0:
            self.weights_gradient[0] = self.weights_gradient[0] + self.lmbd*self.weights[0]

        self.weights[0] = self.weights[0] - self.eta*self.weights_gradient[0]
        self.biases[0] = self.biases[0] - self.eta*self.biases_gradient[0]

        #print(i)
        return error

    def predict(self, X):
        z = [np.zeros(i.shape) for i in self.biases]
        a = [np.zeros(j.shape) for j in self.biases]


        z[0] = np.dot(X,self.weights[0])+self.biases[0]
        a[0] = self.activation(self.activations[0],z[0])

        for i in range(1,len(self.biases)):
            z[i] = np.dot(a[i-1],self.weights[i]) + self.biases[i]
            a[i] = self.activation(self.activations[i],z[i])
        a_o = a[-1]

        probabilities = a_o
        return probabilities

    def SGD(self):

        error_hidden = 10*self.error_min
        datapoints = np.arange(self.X_data.shape[0])

        for i in range(self.epochs):
            j = 0

            while j < self.max_iter and self.error_min < error_hidden:

                chosen_datapoints = np.random.choice(
                datapoints, size=self.batches,replace=False)

                self.X = self.X_data[chosen_datapoints]
                self.y = self.y_data[chosen_datapoints]

                self.FeedForward()
                self.BackwardPropogation()
                j+=1

    def MGD(self):

        for i in range(self.epochs):
            j = 0

            while j < self.max_iter and self.error_min < self.error_output:

                chosen_datapoints = np.random.randint(0,len(self.X[0]))

                self.X_data = self.X[chosen_datapoints]
                self.Y_data = self.y[chosen_datapoints]

                self.FeedForward()
                self.BackwardPropogation()
                j+=1
