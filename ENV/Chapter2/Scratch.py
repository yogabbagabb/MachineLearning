'''
Created on May 23, 2016

@author: ahanagrawal
'''

import numpy as np
from sklearn import preprocessing as prep

if __name__ == '__main__':
    a = np.random.rand(5,5)
    print(a)
    print(prep.minmax_scale(a))
    
    # 0.56391324 - 0.0560/(0.9158 - 0.0560)