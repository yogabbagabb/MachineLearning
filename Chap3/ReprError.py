'''
Created on Jun 7, 2016

@author: ahanagrawal
'''

import numpy as np
import scipy.optimize as sp
import RegLogRegr as reg


def costFunction(theta, X, Y, lam):
    regTerm = lam/(2 * len(X)) * np.dot(theta[1:], theta[1:])
    out = np.zeros(28).T.flatten()
    
    
    return regTerm, out

if __name__ == '__main__':
    '''data = np.loadtxt("ex2/ex2data2.txt", delimiter = ",")
    X,Y = np.split(data, [len(data[0,:]) - 1], 1)
    X = reg.constructVariations(X, 6)
    Y = Y.flatten()
    
    oneArray = np.ones((len(X), 1))
    X = np.hstack((oneArray, X))
    '''
    
    X = np.zeros(118)
    Y = np.zeros(118)
    
    theta = np.zeros(28)
    x = (sp.minimize(costFunction, x0 = theta.T, args = (X,Y, 0.2), method = 'bfgs', jac = True))
    print(x)