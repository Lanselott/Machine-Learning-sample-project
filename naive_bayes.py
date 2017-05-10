
from numpy import *
import numpy as np
import math

class NaiveBayes:
    def __init__(self):
        self.cov_0 = np.mat([1])
        self.cov_1 = np.mat([1])
        self.mean_0 = np.mat([1])
        self.mean_1 = np.mat([1])
        self.num_0 = 0
        self.probability_0 = 0

    def fit(self, X, y):
        d = np.shape(X)[1]  # features
        N = np.shape(X)[0]  # samples
        self.cov_0 = np.zeros(shape=(d,1))
        self.cov_1 = np.zeros(shape=(d,1))
        self.mean_0 = np.zeros(shape=(d,1))
        self.mean_1 = np.zeros(shape=(d,1))
        self.num_0 = np.sum(1.0 if y[n] == 0 else 0.0 for n in range(N))
        self.probability_0 = self.num_0 / N
        tmp_0 = list()
        '''
        sum_0 = np.zeros(shape=(d,1))
        sum_1 = np.zeros(shape=(d,1))

        for i in range (N):
            for j in range (d):
                if(y[i] == 0):
                    sum_0[j] += X[i, j]
                else:
                    sum_1[j] += X[i, j]
        '''
        sum_0 = np.sum(X[n] if y[n] == 0 else 0.0 for n in range(N))
        sum_1 = np.sum(X[n] if y[n] == 1 else 0.0 for n in range(N))

        self.mean_0 = sum_0 / self.num_0
        self.mean_1 = sum_1 / (N-self.num_0)

        self.cov_0 = np.zeros(d)
        self.cov_1 = np.zeros(d)
        for j in range(0, d):
            for i in range(0, N):
                if (y[i] == 0):
                    self.cov_0[j] += (self.mean_0[j] - X[i, j]) * (self.mean_0[j] - X[i, j])
                else:
                    self.cov_1[j] += (self.mean_1[j] - X[i, j]) * (self.mean_1[j] - X[i, j])

            self.cov_0[j] /= self.num_0
            self.cov_1[j] /= (N-self.num_0)
        return self



    def predict(self,X):
        N, d = np.shape(X)  # features
        y = np.zeros(N)
        for i in range(0, N):
            y_0 = self.probability_0
            y_1 = 1 - self.probability_0
            ex0 = 0
            ex1 = 0
            for j in range(0, d):
                ex0 -= (X[i, j] - self.mean_0[j]) * (X[i, j] - self.mean_0[j]) / (2.0 * self.cov_0[j])
                ex1 -= (X[i, j] - self.mean_1[j]) * (X[i, j] - self.mean_1[j]) / (2.0 * self.cov_1[j])

            y_0 *= math.exp(ex0)
            y_1 *= math.exp(ex1)
            for j in range(0, d):
                y_0 /= (math.sqrt(2.0 * math.pi) * math.sqrt(self.cov_0[j]))
                y_1 /= (math.sqrt(2.0 * math.pi) * math.sqrt(self.cov_1[j]))

            if (y_0 > y_1):
                y[i] = 0
            else:
                y[i] = 1

        return y
