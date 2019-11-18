import numpy as np
"""
class NeuralNetwork:
    def __init__(
            self,
            X_data,
            Y_data,
            hidden_neurons=50,
            categories=10,
            epochs=10,
            batches=100,
            max_iter = 100,
            error_min=1e-6,
            eta=0.1,
            lmbd=0.0,
            alpha=1.0,):

        self.X_data_full = X_data
        self.Y_data_full = Y_data

        self.inputs = X_data.shape[0]
        self.features = X_data.shape[1]
        self.hidden_neurons = hidden_neurons
        self.categories = categories

        self.epochs = epochs
        self.batches = batches
        self.max_iter = max_iter #self.inputs // self.batches
        self.eta = eta
        self.lmbd = lmbd
        self.alpha = alpha
        self.error_min = error_min

        self.create_biases_and_weights()

    def create_biases_and_weights(self):
        self.hidden_weights = np.random.randn(self.features, self.hidden_neurons)
        self.hidden_bias = np.zeros(self.hidden_neurons) + 0.01

        self.output_weights = np.random.randn(self.hidden_neurons, self.categories)
        self.output_bias = np.zeros(self.categories) + 0.01

    def activation(self, z_h):
        #print(np.shape(z_h))
        A = 1/(1+np.exp(-z_h))
        #A = np.where(z_h<0, self.alpha*np.exp(z_h)-1, z_h)
        return A

    def softmax(self, z_h):
        #print(np.shape(z_h))
        S = 1/(1+np.exp(-z_h))
        #S = 100*np.exp(z_h)/np.sum(np.exp(z_h))
        return S

    def FeedForward(self):
        # feed-forward for training
        self.z_h = np.dot(self.X_data, self.hidden_weights) + self.hidden_bias
        self.a_h = self.activation(self.z_h)

        self.z_o = np.dot(self.a_h, self.output_weights) + self.output_bias

        exp_term = np.exp(self.z_o)
        self.probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)

    def FFOutput(self, X):
        # feed-forward for output
        z_h = np.dot(X, self.hidden_weights) + self.hidden_bias
        a_h = self.softmax(z_h)

        z_o = np.dot(a_h, self.output_weights) + self.output_bias

        exp_term = np.exp(z_o)
        probabilities = exp_term / np.sum(exp_term, axis=1, keepdims=True)
        return probabilities

    def BackwardPropogation(self):
        error_output = self.probabilities - self.Y_data
        error_hidden = np.dot(error_output, self.output_weights.T) * self.a_h * (1 - self.a_h)

        #print(np.max(error_hidden),np.max(error_output))

        self.output_weights_gradient = np.dot(self.a_h.T, error_output)
        self.output_bias_gradient = np.sum(error_output, axis=0)

        self.hidden_weights_gradient = np.dot(self.X_data.T, error_hidden)
        self.hidden_bias_gradient = np.sum(error_hidden, axis=0)

        if self.lmbd > 0.0:
            self.output_weights_gradient += self.lmbd * self.output_weights
            self.hidden_weights_gradient += self.lmbd * self.hidden_weights

        self.output_weights -= self.eta * self.output_weights_gradient
        self.output_bias -= self.eta * self.output_bias_gradient
        self.hidden_weights -= self.eta * self.hidden_weights_gradient
        self.hidden_bias -= self.eta * self.hidden_bias_gradient
        return error_hidden

    def predict(self, X):
        probabilities = self.FFOutput(X)
        return np.argmax(probabilities, axis=1)

    def predict_probabilities(self, X):
        probabilities = self.FFOutput(X)
        return probabilities

    def SGD(self):

        error_hidden = 10*self.error_min
        datapoints = np.arange(self.inputs)

        for i in range(self.epochs):
            j = 0

            while j < self.max_iter and self.error_min < error_hidden:

                chosen_datapoints = np.random.choice(
                datapoints, size=self.batches,replace=False)

                self.X_data = self.X_data_full[chosen_datapoints]
                self.Y_data = self.Y_data_full[chosen_datapoints]

                self.FeedForward()
                self.BackwardPropogation()
                j+=1
"""
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
            return tanh(z)
        elif activations == "sigmoid":
            return 1/(1+np.exp(-z))
        elif activations == "ReLu":
            return z #temp
        elif activations == "softmax":
            return np.exp(z)/np.sum(np.exp(z), axis=1, keepdims=True)
        else:
            return np.exp(z)/np.sum(np.exp(z), axis=1, keepdims=True)


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
        error_output = self.a_o - self.y
        error[-1] = error_output

        #self.weights_gradient[-1] = np.dot(self.a_o.T, error[-1])
        #self.biases_gradient[-1] = np.sum(error[-1], axis=0)

        #self.weights[-1] = self.weights[-1] - self.eta*self.weights_gradient[-1]
        #self.biases[-1] = self.weights[-1] - self.eta*self.biases_gradient[-1]

        #print(len(self.biases))
        for i in range(len(self.biases)-1,0,-1):
            error[i-1] = np.dot(error[i],self.weights[i].T)*(self.a[i-1]*(1-self.a[i-1]))
            self.biases_gradient[i] = np.sum(error[i], axis=0)
            self.weights_gradient[i] = np.dot(self.a[i-1].T,error[i])

            if self.lmbd > 0.0:
                self.weights_gradient[i] += self.lmbd*self.weights[i]

            #print(np.shape(self.weights_gradient[i]))
            self.weights[i] -= self.eta*self.weights_gradient[i]
            self.biases[i] -= self.eta*self.biases_gradient[i]

        #self.weights_gradient[0] = np.dot(self.X.T, error[0])
        #self.biases_gradient[0] = np.sum(error[0], axis=0)

        #self.weights[0] = self.weights[0] - self.eta*self.weights_gradient[0]
        #self.biases[0] = self.weights[0] - self.eta*self.biases_gradient[0]


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
        return np.argmax(probabilities, axis=1)

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

    def SGD(self):

        error_hidden = 10*self.error_min

        for i in range(self.epochs):
            j = 0

            while j < self.max_iter and self.error_min < error_hidden:

                chosen_datapoints = np.random.randint(0,len(self.X[0]))

                self.X_data = self.X[chosen_datapoints]
                self.Y_data = self.y[chosen_datapoints]

                self.FeedForward()
                self.BackwardPropogation()
                j+=1

"""
np.random.seed(42)
n = 4
X1 = np.zeros((n+2,n))
X2 = np.random.randn(n+2,n)
X = X1 - X2
y = np.random.randn(1,len(X[0]))

print(X,y)
print(np.shape(X))
print(len(X[0]),len(y[0]))


layers = [len(X[0]), 3, 2, 3, len(y[0])]
activations = ["sigmoid","sigmoid","sigmoid","softmax"]
#print(len(activations))
dnn = NeuralNetwork(X,y,layers, activations
)

print(dnn.biases, len(dnn.biases), len(dnn.activations))
print(dnn.weights)
print(dnn.X)
print(dnn.y)
print(dnn.activations)

dnn.SGD()
"""
"""
np.random.seed(42)
n = 4
X1 = np.zeros((n+2,n))
X2 = np.random.randn(n+2,n)
X = X1 - X2
y = np.random.randn(1,len(X[0]))

print(X,y)
print(np.shape(X))
print(len(X[0]),len(y[0]))


layers = [len(X[0]), 3, 2, 3, len(y[0])]
activations = ["sigmoid","sigmoid","sigmoid","softmax"]
#print(len(activations))
dnn = NeuralNetwork(X,y,layers, activations
)

print(dnn.biases, len(dnn.biases), len(dnn.activations))
print(dnn.weights)
print(dnn.X)
print(dnn.y)
print(dnn.activations)

dnn.SGD()
"""

"""
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
            return self.alpha*z
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
            return self.alpha
        elif activations == "softmax":
            return a


    def FeedForward(self):
        # feed-forward for training
        self.z = [np.zeros(i.shape) for i in self.biases]
        self.a = [np.zeros(j.shape) for j in self.biases]

        z[0] = np.dot(self.weights[0].T,self.X) + self.biases
        a[0] = self.activation(self.activations[0],self.z[0])

        for i in range(1,len(self.biases)):
            self.z[i] = np.dot(self.weights[i].T,self.a[i-1]) + self.biases[i]
            self.a[i] = self.activation(self.activations[i],self.z[i])

        self.a_o = self.a[-1]


    def BackwardPropogation(self):
        self.biases_gradient = [np.zeros(i.shape) for i in self.biases]
        self.weights_gradient = [np.zeros(j.shape) for j in self.weights]

        error = np.zeros(i.shape) for i in self.biases
        error_output = self.a_0 - self.y
        error[-1] = error_output

        for i in range(len(self.biases)-1, 0, -1):
            error[i-1] = np.dot(self.weights[i],error[i])*self.activation_derivative(self.activations[i-1],self.a[i-1])
            self.biases_gradient[i] = np.sum(error[i], axis=0)
            self.weights_gradient[i] = np.dot(error[i],self.a[i-1].T)

            if self.lmbd > 0.0:
                self.weights_gradient[i] += self.lmbd*self.weights[i]

            self.weights[i] -= self.eta*self.weights_gradient[i]
            self.biases[i] -= self.eta*self.biases_gradient[i]

        self.weights_gradient[0] = np.dot(error[0],self.X.T)
        self.biases_gradient[0] = np.sum(error[0], axis=0)

        if self.lmbd > 0.0:
            self.weights_gradient[0] = self.weights_gradient[0] + self.lmbd*self.weights[0]

        self.weights[0] = self.weights[0] - self.eta*self.weights_gradient[0]
        self.biases[0] = self.biases[0] - self.eta*self.biases_gradient[0]

        return error


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
"""
"""
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
        error_output = self.a_o - self.y
        error[-1] = error_output
        print(np.shape(self.weights[0]))
        #print(len(self.biases))
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
"""
