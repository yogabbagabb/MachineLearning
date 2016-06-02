'''
Created on May 28, 2016

@author: ahanagrawal
'''

import numpy as np
import matplotlib.pyplot as plt

def segAdmits(D):
    
    admitN = np.sum(D[:,2], axis = 0)
    
    Admits = np.empty((admitN,2))
    Rejects = np.empty((len(D) - admitN,2))
    
    Ai = 0
    Ri = 0
    i = 0
    
    while i < len(D):
        if (D[i,2] == 0):
            Rejects[Ri, 0:2] = D[i, 0:2]
            Ri += 1
        else:
            Admits[Ai, 0:2] = D[i, 0:2]
            Ai += 1
        i += 1
        
    return Admits, Rejects



if __name__ == '__main__':
    data = np.loadtxt("ex2/ex2data1.txt", delimiter = ",")
    Admits, Rejects = segAdmits(data)
    
    plt.scatter(Admits[:,0], Admits[:,1], marker = "+")
    plt.scatter(Rejects[:,0], Rejects[:,1], c = "y", marker = "o")
    plt.xlabel('Test II Scores', fontsize = 12)
    plt.ylabel('Test I Scores', fontsize = 12)
    
    x = np.linspace(0,100,100)
    y = (-1/0.20147107)*(-25.16125626 + 0.20623101*x)
    print(y)
    plt.plot(x,y)
    plt.show()
    
    