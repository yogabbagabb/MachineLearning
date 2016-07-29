'''
Created on Jul 25, 2016

@author: ahanagrawal
'''

import numpy as np
import scipy.io as sio
from scipy.special import expit
import numpy.ma as ma

from data import getData


thetaO, thetaT, X, Y = getData()


'''
Converts the array of ouputs with entry i to a
2 dimensional array in which entry is an array with all entries
but entry i set to 0 and i set to 1. For example, [10, ... 9, ...] becomes
[[1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0]
'''
def constructY(Y, width):
    
   
    Y = ma.masked_equal(Y, value = 10, copy = True)
    ma.set_fill_value(Y,fill_value = 0)
    Y = Y.filled()
    
    Ynew = np.zeros((len(Y), width))
    for i in range(len(Y)):
        Ynew[i][Y[i]] = 1
    return Ynew
    
'''
Returns the cost function evaluated at the current
'''
def cost(X, thetaO, thetaT):
    Ynew = constructY(Y, len(thetaT))
    X = np.matrix(X)
    secLayer = expit(X * thetaO.T)
    onesArray = np.ones((len(secLayer),1))
    secLayer = np.hstack((onesArray, secLayer))
    thrLayer = expit(secLayer * thetaT.T)
    
    m = len(thrLayer)
    k = thrLayer.shape[1]
    cost = 0
    
    for i in range(m):
        for j in range(k):
            cost += -Ynew[i,j]*np.log(thrLayer[i,j]) - (1 - Ynew[i,j])*np.log(1 - thrLayer[i,j])
    print(cost)
    cost /= m
    
    '''
    Regularized Cost Component
    '''
    
    regCost = 0
    
    for i in range(len(thetaO)):
        for j in range(len(thetaO[0])):
            regCost += thetaO[i,j]**2
            
    for i in range(len(thetaT)):
        for j in range(len(thetaT[0])):
            regCost += thetaT[i,j]**2
    
    regCost /= (2*m) 
            
    
    print(cost)
    print(regCost)
    
        
if __name__ == '__main__':
    cost(X,thetaO,thetaT)