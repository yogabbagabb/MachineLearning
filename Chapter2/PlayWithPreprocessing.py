'''
Created on May 24, 2016

@author: ahanagrawal
'''

from sklearn import preprocessing as prep
import numpy as np


if __name__ == '__main__':
    x = np.loadtxt("ex1/ex1data1.txt", delimiter = ",")
    machine = prep.StandardScaler().fit(x)
    print(machine.mean_)
    print(machine.scale_)
    print(machine.transform(x))