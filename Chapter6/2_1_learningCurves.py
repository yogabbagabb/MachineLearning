'''
Created on Aug 1, 2016

@author: ahanagrawal
'''

import numpy as np
import scipy.io as sio
from scipy.stats import linregress

import matplotlib.pyplot as plt


'''
Regularized linear regression was implemented successfully in Chapter2, so we use the
library function in scipy for this exercise. 
'''

x = sio.loadmat("machine-learning-ex5/ex5/ex5data1.mat")
X = x.get("X")
Y = x.get("y")
Xval = x.get("Xval")
Xval = np.hstack((np.ones(len(Xval),1), Xval))

Yval = x.get("yval")
Xtest = x.get("Xtest")
Ytest = x.get("ytest")

def costFunction(x, y, theta):
    sum = 0
    for i in range(len(x)):
        sum += (np.dot(x[i,:],theta) - y[i])**2
    return (sum/(len(x) *2))

if __name__ == '__main__':
    
    stderr = np.matrix(np.zeros((len(X),1)))
    stderr_cVal = np.matrix(np.zeros((len(X),1)))
    
    
    for i in range(1,len(X)):
        print(np.cov(X[0:i,:], Y[0:i,:]))
        
    for i in range(1,len(X)):
        slope, intercept, r_value, p_value, std_err = (linregress(X[0:i,:], y = Y[0:i,:]))
        theta = np.array([intercept, slope])
        stderr[i] = std_err
        stderr_cVal[i] = costFunction(Xval, Yval, theta)
         
    plt.plot(X, stderr)
    plt.plot(Xval, stderr_cVal)
     
    plt.show()
#     
        
         
   
    
    