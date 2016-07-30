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
     
    d3 = expit(1 + 2*expit(15) + 3*expit(31))
    a = d3 - 8
    print(a)
    q = np.asarray([1,2,3])
    r = np.asarray([2,3,4])
    
    print((a)*expit(1) * (1-expit(1)), 2*(a)*expit(15)*(1-expit(15)), 3*(a)*expit(31)*(1-expit(31)))
   

        