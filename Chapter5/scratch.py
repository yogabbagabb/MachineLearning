'''
Created on Jul 27, 2016

@author: ahanagrawal
'''

import numpy as np

def testPassByReference(array):
    
    array = [1,2,3,4]
    array[2] += 2
    print(array)

if __name__ == '__main__':
    a = np.array([[1,2,3],[4,5,6]])
    print(len(a[0]))
    b = np.asmatrix(a)
    print(b.shape[1])
        