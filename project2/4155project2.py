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
#import tensorflow as tf

def activation(x, type, c=None):
    if type == 'sigmoid':
        return 1/(1+np.exp(-x))
    elif type == 'tanh':
        return c*tanh(x)
    elif type == 'softmax':
        return 1/(1+np.exp(-x))
    elif type == 'ELU':
        return a*np.exp(x)-1 if x < 0 else x


def GradientDecent(f, C0, gamma, eps=1e-5, max_iter):
    """
    returns...
    """
    i = 0
    C = C0
    while i < max_iter or C < eps:
        C = gamma
        i+=1

    return grad

def FeedForward(X):
    return 1

def BackwardPropogation(X,Y):
    return 1


def main():
    """
    something about what to do here
    """
    #hyperparameters

if __name__ == "__main__":
    main()
