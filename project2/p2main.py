import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
#from p2NN import *
from NN import *
from sklearn.metrics import accuracy_score
import seaborn as sns

def to_categorical_numpy(integer_vector):
    n_inputs = len(integer_vector)
    categories = np.max(integer_vector) + 1
    onehot_vector = np.zeros((n_inputs, categories))
    onehot_vector[range(n_inputs), integer_vector] = 1

    return onehot_vector

# ensure the same random numbers appear every time
np.random.seed(42)


# display images in notebook
plt.rcParams['figure.figsize'] = (12,12)


# download MNIST dataset
digits = datasets.load_digits()

# define inputs and labels
inputs = digits.images
labels = digits.target

n_inputs = len(inputs)
inputs = inputs.reshape(n_inputs, -1)

print("inputs = (n_inputs, pixel_width, pixel_height) = " + str(inputs.shape))
print("labels = (n_inputs) = " + str(labels.shape))
print("X = (n_inputs, n_features) = " + str(inputs.shape))
print(inputs[0])

# flatten the image
# the value -1 means dimension is inferred from the remaining dimensions: 8x8 = 64



# choose some random images to display
indices = np.arange(n_inputs)
random_indices = np.random.choice(indices, size=5)

# one-liner from scikit-learn library
train_size = 0.8
test_size = 1 - train_size
X_train, X_test, Y_train, Y_test = train_test_split(inputs, labels, train_size=train_size,
                                                    test_size=test_size)


for i, image in enumerate(digits.images[random_indices]):
    plt.subplot(1, 5, i+1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title("Label: %d" % digits.target[random_indices[i]])
plt.show()

#Y_train_onehot, Y_test_onehot = to_categorical(Y_train), to_categorical(Y_test)
Y_train_onehot, Y_test_onehot = to_categorical_numpy(Y_train), to_categorical_numpy(Y_test)


print("Number of training images: " + str(len(X_train)))
print("Number of test images: " + str(len(X_test)))

epochs = 20
batch_size = 100
max_iter = 500
error_min = 0.01
eta_vals = np.logspace(-5, 0, 6)
lmbd_vals = np.logspace(-3, -1, 8)
#alpha_vals = np.logspace(-5, 1, 7)
# store the models for later use
DNN_numpy = np.zeros((len(eta_vals), len(lmbd_vals)), dtype=object)
print(len(X_train[0]))
layers = [len(X_train[0]), 100, 50, len(Y_train_onehot[0])]
activations = ["sigmoid","sigmoid","softmax"]#,"sigmoid","sigmoid","softmax"]
#print(len(activations))

# grid search
for i, eta in enumerate(eta_vals):
    for j, lmbd in enumerate(lmbd_vals):
        alpha = 0.1
        dnn = NeuralNetwork(X_train,Y_train_onehot,layers, activations,
        epochs=epochs, batches=batch_size, max_iter=max_iter,
        error_min=error_min, eta=eta, lmbd=lmbd, alpha=alpha)

        #dnn = NeuralNetwork(X_train, Y_train_onehot)#, hidden_neurons=hidden_neurons, categories=categories, learningrate=learningrate, lmbd=lmbd, epochs=epochs, batches=batches, max_iter=max_iter, error_min=error_min)
        dnn.SGD()
        test_predict = np.argmax(dnn.predict(X_test), axis=1)

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

#dnn = NeuralNetwork(X_train, Y_train_onehot)#, hidden_neurons=hidden_neurons, categories=categories, learningrate=learningrate, lmbd=lmbd, epochs=epochs, batches=batches, max_iter=max_iter, error_min=error_min)
#self, X_in, y_in, hidden_neurons=20, categories=10, learningrate=0.1, lmb=0, epochs=5, batches=50, max_iter=1e3, error_min=0
#dnn.SGD()
#test_predict, prediction = dnn.predictions(X_test)
#print(prediction)
