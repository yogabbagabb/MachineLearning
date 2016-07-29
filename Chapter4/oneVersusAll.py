'''
Created on Jun 13, 2016

@author: ahanagrawal
'''

import scipy.io as sio
import numpy as np
from sklearn.linear_model import LogisticRegression as expit


if __name__ == '__main__':
    data = sio.loadmat("machine-learning-ex3/ex3/ex3data1.mat")
    X = data.get('X')
    Y = data.get('y')
    oneArray = np.ones((len(X),1))
    X = np.hstack((oneArray, X))
    theta = np.zeros(shape = (10, len(X[0])))
     
    np.save("possibleX.npy", X)
    np.save("possibleY.npy", Y)
 
    print("done")
    
    key = 20 
    for i in range(10):
        if (i == 0):
            key = 10
        else:
            key = i
        
        yTemp = np.ma.masked_not_equal(Y, key).filled(0)    
        yTemp = np.ma.masked_equal(yTemp, key).filled(1)    
        
        
        trial = expit(solver = 'newton-cg')
        trial = expit.fit(trial, X = X, y = yTemp.ravel())
        theta[i][:] = trial.coef_
    
    theta = np.asmatrix(theta)
    np.save("possibleTheta.npy", theta)
    X = np.asmatrix(X)
    suc = 0
    att = 0
    
    while att < len(X):
        output = theta * X[att][:].T
        l = np.argmax(output, 0)
        print(l)
        att += 1
                
    