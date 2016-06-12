'''
Created on Jun 6, 2016

@author: ahanagrawal
'''

import scipy.io as sio

if __name__ == '__main__':
    X = sio.loadmat("machine-learning-ex3/ex3/ex3data1.mat")
    print(X.get('y'))
#     print(X.y[0], X.y[40], X.y[100])
    