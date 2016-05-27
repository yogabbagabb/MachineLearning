'''
Created on May 22, 2016

@author: ahanagrawal
'''


from sklearn import preprocessing as prep
import numpy as np
import pylab

def descend(X, Y, theta, alpha, tolerance):
    
    cost = costFunction(X, Y, theta)
    
    while True:
        
        thCopy = np.copy(theta)
        
        for i in range(len(thCopy)):
            sum = 0
            for j in range(len(thCopy)):
                sum += ((np.dot(theta, X[i])) - Y[i])*X[i,j]
            thCopy[i] = theta[i] - alpha*(1/len(X))*sum
        
        newCost = costFunction(X, Y, thCopy)
        if (abs(cost - newCost) < tolerance):
            break
        else:
            theta = thCopy
            cost = newCost
            print(cost)
    
    return thCopy
    

def costFunction(x, y, theta):
    sum = 0
    for i in range(len(x)):
        sum += (np.dot(x[i,:],theta) - y[i])**2
    return (sum/(len(x) *2))

def ConstructArrays(array):
    shape = np.shape(array)
    width = shape[1]
    onesArray = np.ones((len(array),1))
    splitArrays = np.split(array, [width - 1], 1)
    
    splitArrays[0] = prep.minmax_scale(splitArrays[0])
    featuresArray = np.hstack([onesArray, splitArrays[0]])
    outputArray = splitArrays[1] 
    'prep.minmax_scale(splitArrays[1])' 
    
    
    
    
    return featuresArray, outputArray

    

if __name__ == '__main__':
    array = pylab.loadtxt("ex1/ex1data1.txt", dtype = float, delimiter = ",")
    features, output, min, max = ConstructArrays(array)
    
    shape = np.shape(array)
    width = shape[1]
    theta = np.zeros(width)

    print(costFunction(features, output, [-5.2, 1.5]))
    
    theta = descend(features, output, theta, 0.1, 0.0001)
    print(theta)