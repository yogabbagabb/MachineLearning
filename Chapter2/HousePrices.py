'''
Created on May 24, 2016

@author: ahanagrawal
'''

import numpy as np
import matplotlib.pyplot as plt
from GradientDescent import *


def main(*args):
    D = np.loadtxt("ex1/ex1data2.txt", delimiter = ",")
    X, Y = ConstructArrays(D)
    X = np.mat(X)
    Y = np.mat(Y)
    theta = np.linalg.inv(X.T * X) * X.T * Y
    
    'print(theta)'
    return X,Y

if __name__ == '__main__':
    main()
    
    