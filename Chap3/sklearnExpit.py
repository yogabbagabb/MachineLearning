'''
Created on Jun 3, 2016

@author: ahanagrawal
'''

from sklearn.linear_model import LogisticRegression as expit
import numpy as np
import RegLogRegr as reg

if __name__ == '__main__':
    data = np.loadtxt("ex2/ex2data2.txt", delimiter = ",")
    X,Y = np.split(data, [len(data[0,:]) - 1], 1)
    X = reg.constructVariations(X, 6)
    
    oneArray = np.ones((len(X),1))
    X = np.hstack((oneArray, X))
    trial = expit(solver = 'sag')
    trial = trial.fit(X = X,y = np.ravel(Y))
    print(trial.coef_)