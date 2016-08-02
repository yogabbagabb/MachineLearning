'''
Created on Aug 1, 2016

@author: ahanagrawal

'''

from scipy import stats
import numpy as np


if __name__ == '__main__':
    x = np.random.random(10)
    y = np.random.random(10)
    print(x)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)