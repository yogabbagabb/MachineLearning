'''
Created on Jul 26, 2016

@author: ahanagrawal
'''
from scipy.special import expit 
from numpy.random import rand as randArr
import numpy as np

import scipy.io as sio
import numpy.ma as ma
import csv

from data import getData




'''
Converts the array of ouputs with entry i to a
2 dimensional array in which entry is an array with all entries
but entry i set to 0 and i set to 1. For example, [10, ... 9, ...] becomes
[[1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0]
'''
def constructY(Y, width):
    
#    
#     Y = ma.masked_equal(Y, value = 10, copy = True)
#     ma.set_fill_value(Y,fill_value = 0)
#     Y = Y.filled()
    
    Ynew = np.zeros((len(Y), width))
    for i in range(len(Y)):
        Ynew[i][Y[i]-1] = 1
    return Ynew
    
'''
Returns the cost function evaluated at the current
'''
def cost(X, Y, thetaO, thetaT, lam):
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
#     print(cost)
    cost /= m
    
    '''
    Regularized Cost Component
    '''
    
    regCost = 0
    
    for i in range(len(thetaO)):
        for j in range(1,(thetaO.shape[1])):
            regCost += thetaO[i,j]**2
            
    for i in range(len(thetaT)):
        for j in range(1,(thetaT.shape[1])):
            regCost += thetaT[i,j]**2
    
    regCost *= lam/(2.*m) 
            

#     print(cost)
#     print(regCost)
    
    return cost + regCost
'''
Returns the derivative of the expit function evaluated at z

Parameters
____

z: float
        Any real number
        
Example
____

>>> sigprime(0)
0.25
'''
def sigprime(z):
    return np.multiply(expit(z),(1-expit(z)))


''' Constructs randomly weighted neural networks based on the desired architecture of the network.

Parameters
____

layerDim: tuple
        A tuple of ints that must contain at least two entries. If the entries are m in number, then
        the first array of weights has shape(entry_1, entry_2), the second array has shape (entry_2, entry_3),
        and so on, until the final array has shape (entry_m-1, entry_m)

Returns
____
layerWeights: list
        A list of the network weights between each set of layers.

Examples
____
randInit(2,4,5) returns a tuple consisting of two randomly initialized arrays, the first
with shape(2,4) and the second with shape (4,5).

>>> randInit(2,4,5)
[array([[-0.29998689, -0.61859535, -0.58762326,  0.78933139],
       [-0.41674141,  0.41700764, -0.06658324,  0.17097864]]), array([[ 0.43857363,  0.57714546,  0.19039167, -0.71508532,  0.06942178],
       [ 0.56639229, -0.54119974,  0.26075461,  0.69557778,  0.52747973],
       [ 0.49458375,  0.07569438,  0.71165007,  0.17264876,  0.02798658],
       [ 0.11826181, -0.04742607,  0.77680679,  0.21094343, -0.0938325 ]])]


'''
def randInit(layerDim):
    layerWeights = []
    
    for i in range(len(layerDim) - 1):
        x = layerDim[i]
        y = layerDim[i+1]
        epsilon = np.sqrt(6)/np.sqrt(x + y)
        layerWeights.append(randArr(x,y) * 2*epsilon - epsilon)
    return layerWeights
    
'''
Forward propagates through a 3 layer network.

Parameters
____

X: numpy array
        An array consisting of any number of trials. X is assumed to have
        its column of 1s (i.e the bias factors) appended already
thetaO: numpy array
        A weight array from layer (1) --> layer(2)
thetaT: numpy array
        A weight array from layer (2) --> layer(3)


Returns
____
thrLayer: numpy array
        The output layer.

Examples
____
See the course sheet under ex3.pdf in Chapter4/machine-learning-ex3

Notes
____


Note
____
There are several ways to define thetaO and thetaT. This function
assumes that these weight arrays have defined in such a way that
thetaThr = thetaT * thetaO * X

where every column of X constitutes one trial:

t t t t t
r r r r r 
i i i i i 
a a a a a
l l l l l

Alternatively

thetaThr.T = X.T * thetaO.T * thetaT.T

if we imagine that every row of X constitutes one trial:

t r i a l
t r i a l
t r i a l
t r i a l
'''
def forwardProp(X, thetaO, thetaT):
    
    X = np.matrix(X)
    secLayer = expit(X * thetaO.T)
    onesArray = np.ones((len(secLayer),1))
    secLayer = np.hstack((onesArray, secLayer))
    thrLayer = expit(secLayer * thetaT.T)    
    
    return thrLayer, secLayer

'''
Validates whether the gradient, as approximated by a difference equation in the Cost Function,
is the same as the gradient as approximated by backpropagation. See lecture videos from chapter 5
for further explanation
'''
def checkGradient(X, Y, thetaO, thetaT):
    
    epsilon = 10E-4
    
    DeltaOne, DeltaTwo = backprop(X, Y, thetaO, thetaT, 1)
    
    DeltaOneDiff = np.zeros(np.shape(DeltaOne))
    DeltaTwoDiff = np.zeros(np.shape(DeltaTwo))
    
    for i in range(len(thetaO)):  
        for j in range(thetaO.shape[1]):
            
            thetaOFirst = np.copy(thetaO)
            thetaOSecond = np.copy(thetaO)
            
            thetaOFirst[i,j] += epsilon
            thetaOSecond[i,j] -= epsilon
            
            DeltaOneDiff[i,j] = (cost(X, Y, thetaOFirst, thetaT, 1) - cost(X, Y, thetaOSecond, thetaT, 1)) / (2*epsilon)
    
    for i in range(len(thetaO)):  
        for j in range((thetaO.shape[1])):
            
            thetaTFirst = np.copy(thetaT)
            thetaTSecond = np.copy(thetaT)
            
            thetaTFirst[i,j] += epsilon
            thetaTSecond[i,j] -= epsilon
            
            DeltaTwoDiff[i,j] = (cost(X, Y, thetaO, thetaTFirst, 1) - cost(X, Y, thetaO, thetaTSecond, 1)) / (2*epsilon)
    
    DeltaOneDiff = np.abs(DeltaOneDiff - DeltaOne) 
    DeltaTwoDiff = np.abs(DeltaTwoDiff - DeltaTwo)
    
    print("Error detected: ", DeltaOneDiff > 0.0001)


def backprop(X, Y, thetaO, thetaT, lam):
    thrLayer, secLayer = forwardProp(X, thetaO, thetaT)
    yNew = constructY(Y, len(thetaT))
    
    m = len(X)
    DeltaOne = np.zeros(np.shape(thetaO))
    DeltaTwo = np.zeros(np.shape(thetaT))
    
    for i in range(m):
        d3 = thrLayer[i] - yNew[i]
        d2 = np.multiply(thetaT.T*d3.T, sigprime(secLayer[i,:].T))
        DeltaTwo += d3.T * expit(secLayer[i,:])
        a = d2[1:]
        b = np.asmatrix((expit(X[i,:])))
        DeltaOne += a*b 

    m = float(m)        
    DeltaOne /= m
    DeltaTwo /= m
    
    DeltaOne[0:,1:] += thetaO[0:,1:]*lam/m
    DeltaTwo[0:,1:] += thetaT[0:,1:]*lam/m
        
    return DeltaOne, DeltaTwo
        
        

thetaO, thetaT, X, Y = getData()

    
if __name__ == '__main__':
    checkGradient(X, Y, thetaO, thetaT)
        
