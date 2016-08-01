'''
Created on Aug 1, 2016

@author: ahanagrawal

'''

from scipy.special import expit 
from numpy.random import rand as randArr
import numpy as np

from scipy import optimize


def forwardProp(X, thetaO, thetaT):
    
    X = np.matrix(X)
    secLayer = expit(X * thetaO.T)
    onesArray = np.ones((len(secLayer),1))
    secLayer = np.hstack((onesArray, secLayer))
    thrLayer = expit(secLayer * thetaT.T)    
    
    return thrLayer, secLayer

def sigprime(z):
    return np.multiply(expit(z),(1-expit(z)))

def constructY(Y, width):
    
#    
#     Y = ma.masked_equal(Y, value = 10, copy = True)
#     ma.set_fill_value(Y,fill_value = 0)
#     Y = Y.filled()
    
    Ynew = np.zeros((len(Y), width))
    for i in range(len(Y)):
        Ynew[i][0] = 1
    return Ynew

def backprop(X, Y, thetaO, thetaT, lam):
    thrLayer, secLayer = forwardProp(X, thetaO, thetaT)
    # thetaO = a 25 x 401 weight matrix
    # thetaT = a 10 x 26 weight matrix
    # thrLayer = a 5000 x 10 matrix; row i contains 10 entries, the greatest of which designates the class
    # secLayer = a 5000 x 26 matrix; the second layer post activation; has a bias unit
    # X = a 5000 x 401 matrix; the input one
    # Y = a 5000 x 1 matrix; row i has a number representing the class (0-9) corresponding to the entry in row i of X
    
    yNew = constructY(Y, len(thetaT))
    #The 5000 x 10 output matrix, each
    
    m = len(X)
    DeltaOne = np.zeros(np.shape(thetaO))
    DeltaTwo = np.zeros(np.shape(thetaT))
    
    for i in range(m):
        d3 = thrLayer[i] - yNew[i]
        # Row vector
        
#         print(thetaT.T * d3.T)
#         print(sigprime(secLayer[i,:].T))
#         print(secLayer[i,:])
#         print(sigprime(1),sigprime(15),sigprime(31))
        
        d2 = np.multiply((thetaT.T*d3.T)[1:], sigprime((X[i,:] * thetaO.T)).T)
        DeltaTwo += d3.T * (secLayer[i,:])
        b = np.asmatrix(((X[i,:])))
        DeltaOne += d2*b 

    m = float(m)        
    DeltaOne /= m
    DeltaTwo /= m
    
    DeltaOne[0:,1:] += thetaO[0:,1:]*lam/m
    DeltaTwo[0:,1:] += thetaT[0:,1:]*lam/m
        
    return DeltaOne, DeltaTwo
        
        

