'''
Created on Jul 27, 2016

@author: ahanagrawal
'''

import numpy as np
from scipy.special import expit
from Chapter5.backprop import sigprime

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
     
    a = np.arange(20)
    a = np.reshape(a[0:10], (5,2), order = 'F')
    print(a)
    b = a.flatten()
    c = np.arange(3).flatten()
    print(b,c)
    print(np.hstack((b,c)))
    print(b)
    print(np.reshape(b, (4,5), order = 'F'))

        