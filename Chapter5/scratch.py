'''
Created on Jul 27, 2016

@author: ahanagrawal
'''

import numpy as np
from scipy.special import expit
from Chapter5.backprop import sigprime, flatToWeight

def testFlatToWeights():
    a = np.arange(20)
    print(a)
    weightDim = (2,4,6,2)
    q = flatToWeight(a, weightDim)
    print(q[1].flatten())

def testPassByReference(array):
    
    array = [1,2,3,4]
    array[2] += 2
    print(array)
    
    
    

def testDimArray():
    a = np.array([[1,2,3],[4,5,6]])
    print(len(a[0]))
    b = np.asmatrix(a)
    print(b.shape[1])
    
if __name__ == '__main__':
     
    testFlatToWeights()

        