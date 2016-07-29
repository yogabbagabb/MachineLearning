'''
Created on May 31, 2016

@author: ahanagrawal
'''

import numpy as np
import scipy.special as sigmoid
import math as m
import scipy.optimize as sp


def constructVariations(X, deg):
    
    features = np.zeros((len(X), 27)) 
    copies = sum
    spot = 0
    stringK = []
    for i in range(1, deg + 1):
        for j in range(i + 1):
            features[:, spot] = X[:,0]**(i - j) * X[:,1]**(j)
            stringK.append("x^" + str(i-j) + "y^" + str(j))
            spot += 1

    return features

def costFunction(theta, X, Y, lam):
    s = 0
    for i in range(len(X)):
        a = np.dot(X[i,:],theta)
        sig = sigmoid.expit(np.dot(X[i,:],theta))
        if (sig == 1):
            sig = 0.999999999999
        if (sig == 0):
            sig = 0.000000000001
        try:
            d = (-1/len(X))*((1 - Y[i])*m.log(1 - sig) + (Y[i])*m.log(sig))
        except ValueError:
            print(sig) 
            
        s = s + d
    
    regTerm = lam/(2 * len(X)) * np.dot(theta[1:], theta[1:])
    
    '''test'''
    
    h = sigmoid.expit(X.dot(theta))

    thetaR = theta[1:].T

    delta = h - Y
    sumdelta = delta.T.dot(X[:, 1])
    grad1 = (1.0 / len(X)) * sumdelta

    XR = X[:, 1:X.shape[1]]
    sumdelta = delta.T.dot(XR)

    grad = (1.0 / len(X)) * (sumdelta + lam * thetaR)

    out = np.zeros(grad.shape[0] + 1)
    out[0] = grad1
    out[1:] = grad
    out = out.T.flatten()
    
    '''test'''
        
    return s + regTerm
        
if __name__ == '__main__':
    data = np.loadtxt("ex2/ex2data2.txt", delimiter = ",")
    X,Y = np.split(data, [len(data[0,:]) - 1], 1)
    X = constructVariations(X, 6)
    Y = Y.flatten()
    
    oneArray = np.ones((len(X), 1))
    X = np.hstack((oneArray, X))
    theta = np.zeros(28)
    print(costFunction(theta.T, X, Y, 0.1))
    x = (sp.minimize(costFunction, x0 = theta.T, args = (X,Y, 0.2), method = 'Powell', jac = None))
    print(x)
    