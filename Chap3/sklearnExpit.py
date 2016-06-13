'''
Created on Jun 3, 2016

@author: ahanagrawal
'''

from sklearn.linear_model import LogisticRegression as expit
import numpy as np
import RegLogRegr as reg

def constructVariations(X, deg):
    
    features = np.zeros((len(X), 27)) 
    copies = sum
    spot = 0
    stringK = []
    for i in range(1, deg + 1):
        for j in range(i + 1):
            features[:, spot] = X[:,0]**(i - j) * X[:,1]**(j)
            stringK.append("x^" + str(i-j) + "y^" + str(j))
            spot += 1

    return features

if __name__ == '__main__':
    data = np.loadtxt("ex2/ex2data2.txt", delimiter = ",")
    X,Y = np.split(data, [len(data[0,:]) - 1], 1)
    X = constructVariations(X, 6)
    
    oneArray = np.ones((len(X),1))
    X = np.hstack((oneArray, X))
    trial = expit(solver = 'sag')
    trial = trial.fit(X = X,y = np.ravel(Y))
    print(trial.coef_)
    print(trial.intercept_)