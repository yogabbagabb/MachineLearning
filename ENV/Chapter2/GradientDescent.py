'''
Created on May 22, 2016

@author: ahanagrawal
'''

from sklearn import preprocessing
import numpy as np
import pylab


def costFunction(x, y, theta):
    sum = 0
    for i in range(len(x)):
        sum += (np.dot(x[i,:],theta) - y[i])**2
    print (sum)

def ConstructArrays(array):
    shape = np.shape(array)
    width = shape[1]
    onesArray = np.ones((len(array),1))
    splitArrays = np.split(array, [width - 1], 1)
    
    
    featuresArray = np.hstack([onesArray, splitArrays[0]])
    outputArray = splitArrays[1]
    
    return featuresArray, outputArray

    

if __name__ == '__main__':
    array = pylab.loadtxt("ex1/ex1data1.txt", dtype = float, delimiter = ",")
    features, output = ConstructArrays(array)
    
    shape = np.shape(array)
    width = shape[1]
    theta = np.zeros( width)
    
    costFunction(features, output, theta)