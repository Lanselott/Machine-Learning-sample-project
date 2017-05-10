import random
import numpy as np
import math
import scipy.optimize as opt
from sklearn import preprocessing



def sigmoid(X):
    '''get sigmoid function'''
    return 1 / (1 + np.exp(-X))


def compute_grad(theta, X, y, m, feature):
    m = np.shape(X)[0]
    theta.shape = (1, feature)
    grad = np.zeros(feature)
    h = sigmoid(X.dot(theta.T))  # 3695*108 dot 108*1
    delta = h - y
    #print (np.shape(delta))
    for i in range(grad.size):
        sumdelta = delta.T.dot(X[:, i])  # 1*3695 dot 3695*1 = 1*1
        #sumdelta = np.dot(np.transpose(delta), (X[:, i]))
        grad[i] = (1.0 / m) * sumdelta * -1

    theta.shape = (feature,)
    return grad



def Gradient_Descent(X, y, theta, m, alpha,feature):
    new_theta = np.zeros(feature)
    for j in range(len(theta)):
        new_grad_dev = compute_grad(theta, X, y, m, feature)
        new_theta_value = theta[j] + new_grad_dev
        new_theta[j] = new_theta_value[j]
    return new_theta


def Logistic_Regression(X, y, alpha, theta, num_iters,feature):
    m = len(y)
    for x in range(num_iters):
        new_theta = Gradient_Descent(X, y, theta, m, alpha,feature)
        theta = new_theta
    return theta


class LogisticRegression:
    def __init__(self):
        self.coef = np.zeros(10)

    def fit(self,X,y,alpha,iterate):
        y = np.reshape(y, (np.shape(y)[0], 1))
        '''regularization the data'''
        X = preprocessing.scale(X)
        '''constant'''
        constant = np.ones((np.shape(X)[0], 1))
        X = np.c_[X, constant]

        N,d = np.shape(X)
        theta = np.zeros(d)
        for i in range (0,d):
            theta[i] = random.uniform(-0.01,0.01)

        #print('parameter',X,y,alpha,iterate,d)
        self.coef = Logistic_Regression(X,y,alpha,theta,iterate,d)
        #print("random",theta)
        return self


    def predict(self, X):
        m, n = X.shape
        p = np.zeros(shape=(m, 1))
        constant = np.ones((np.shape(X)[0], 1))
        X = np.c_[X, constant]
        #print("lr.py coef in predict",self.coef)
        h = sigmoid(X.dot(self.coef.T))
        #print("final theta",self.coef)
        for it in range(0, np.shape(h)[0]):
            if h[it] > 0.5:
                p[it, 0] = 1
            else:
                p[it, 0] = 0

        return p
