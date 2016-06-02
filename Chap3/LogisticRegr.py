'''
Created on May 29, 2016

@author: ahanagrawal
'''

import numpy as np
import math as m
import scipy.optimize as sp
import scipy.special as sigmoid

def anotherF(theta, X, Y):
    return  -costFunction(theta, X, Y)

def costFunction(theta, X, Y):
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
    return s
if __name__ == '__main__':
    data = np.loadtxt("ex2/ex2data1.txt", delimiter = ",")
    X,Y = np.split(data, [len(data[0,:]) - 1], 1)
    
    oneArray = np.ones((len(X),1))
    X = np.hstack((oneArray, X))
    theta = [0.1,0.1,0.1]
    print(costFunction(theta, X, Y))
    print(sp.minimize(costFunction, x0 = theta, args = (X,Y), method = 'BFGS'))
    
