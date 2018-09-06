import numpy as np
import pandas as pd
from patsy import dmatrices

def sigmoid(x):
    return 1/(1 + np.exp(-x))

np.random.seed(0)

tol = 1e-8
max_iter = 20
lam = None

#generate toy data
cov = 0.95 #covariance term
n = 1000 #number of observations
sigma = 1 #stdev

betaX = -4
betaZ = 0.9
betaV = 1

varX = 1
varZ = 1
varV = 4