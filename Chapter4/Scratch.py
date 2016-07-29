'''
Created on Jun 6, 2016

@author: ahanagrawal
'''

import numpy as np
if __name__ == '__main__':
    theta = np.load("possibleTheta.npy")
    X = np.load("possibleX.npy")
    Y = np.load("possibleY.npy")
    
    print("a", theta[3][:])
    print("a", theta[4][:])

    theta = np.asmatrix(theta)
    X = np.asmatrix(X)
    suc = 0
    att = 0
    
    
    
    
    count = np.zeros(10)
    while att < len(X):
        output = theta * X[att][:].T
        output = 1/(1 + np.exp(-output))
        l = np.argmax(output, 0)
        print(output, l)
        count[l] += 1
        
        if l == 0:
            l = 10
        
        if (l == Y[att]):
            suc += 1
        att += 1    
        
        
    print(count)
    
    base = np.zeros(10)
    for i in range(len(Y)):
        key = Y[i]
        if (key == 10):
            key = 0
        base[key] += 1
    print(base)
    print(suc/att)
#     A = Y[450:550]
#     
#     yTemp = np.ma.masked_equal(A, 10).filled(2)    
#     yTemp = np.ma.masked_not_equal(yTemp, 2).filled(0)
#     print(A, yTemp)    