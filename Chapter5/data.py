'''
Created on Jul 25, 2016

@author: ahanagrawal
'''

import numpy as np
import scipy.io as sio

theta = sio.loadmat("ex4/ex4weights.mat")
data = sio.loadmat("ex4/ex4data1.mat")

thetaO = np.matrix(theta.get('Theta1'))
thetaT = np.matrix(theta.get('Theta2'))


X = np.load("possibleX.npy")
Y = np.load("possibleY.npy")

Xprime = np.matrix(data.get('X'))

x = 4

def getData():
    return thetaO, thetaT, X,Y

if __name__ == '__main__':
    pass