'''
Created on Jun 6, 2016

@author: ahanagrawal
'''


import numpy as np

if __name__ == '__main__':
    
    a = np.asarray([1,2,3,4])
    a = np.ma.masked_not_equal(a, 1)
    print(a.filled(0))
    
    b = np.ma.array([1,2,3])
    b.mask = (b < 2) | (b > 2)
    print(b)
        
#     X = sio.loadmat("machine-learning-ex3/ex3/ex3data1.mat")
#     print(X.get('X'))
#     print(X.get('y'))
#     print(X.y[0], X.y[40], X.y[100])
    