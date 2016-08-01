'''
Created on Jul 30, 2016

@author: ahanagrawal
'''
import scipy.io as sio
import numpy as np
from scipy.special import expit
from Chapter5.backprop import flatToWeight

if __name__ == '__main__':
    weights = np.load("solutionWeights.npy")
    weightDim = (25,401,10,26)
    
    thetaO,thetaT = flatToWeight(weights, weightDim)
    
    X = np.load("possibleX.npy")
    Y = np.load("possibleY.npy")
    X = np.matrix(X)
    secLayer = expit(X * thetaO.T)
    onesArray = np.ones((len(secLayer),1))
    secLayer = np.hstack((onesArray, secLayer))
    thrLayer = expit(secLayer * thetaT.T)
    
    for i in range(len(thrLayer)):
        print(np.argmax(thrLayer[i]), Y[i])
   
    
    suc = 0
    trials = 0
    
    while trials < len(X):
        val = np.argmax(thrLayer[trials])
#         if val == 0:
#             val = 10
        if (val + 1 == Y[trials]):
            suc += 1
        trials += 1
    print(suc/trials)
            
        
    