'''
Created on Jun 5, 2016

@author: ahanagrawal
'''
from GradientDescent import ConstructArrays
import pylab
import numpy as np

if __name__ == '__main__':
    array = pylab.loadtxt("ex1/ex1data1.txt", dtype = float, delimiter = ",")
    X,Y = ConstructArrays(array)
    X = np.matrix(X)
    Y = np.matrix(Y)
    print(pylab.inv(X.T*X) * X.T * Y)