'''
Created on Jun 14, 2016

@author: ahanagrawal
'''

import scipy.io as sio
import numpy as np
from scipy.special import expit

if __name__ == '__main__':
    matr = sio.loadmat("machine-learning-ex3/ex3/ex3weights.mat")
    thetaO = np.matrix(matr.get('Theta1'))
    thetaT = np.matrix(matr.get('Theta2'))
    
    
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
            
        
    