'''
Created on May 22, 2016

@author: ahanagrawal
'''

import numpy as np
import pylab

def costFunction(x, y, theta):
    sum = 0
    for i in range(len(x)):
        sum += (np.dot(x[i,:],theta) - y[i])**2
    print (sum)


if __name__ == '__main__':
    array = pylab.loadtxt("ex1/ex1data1.txt", dtype = float, delimiter = ",")
    trials = np.ones((1,len(array) + 1))
    trials[0:len(array)] = array[:,0]
    theta = np.zeros((1,len(array) + 1))
    costFunction(array[:, 0], array[: , 1], theta)