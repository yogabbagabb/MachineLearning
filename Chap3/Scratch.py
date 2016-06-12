'''
Created on May 29, 2016

@author: ahanagrawal
'''

import LogisticRegr as l
import numpy as np
import math
import scipy.special as sigmoid
import re
import csv


import numpy as np
from _csv import Dialect
if __name__ == '__main__':
    '''print(np.r_[np.array([1,2,3]), 0, 1, np.array([4,5,6])])
    print(np.r_['0,2',np.array([[1,2,3]]), np.array([[4,5,6]])])
    '''
    '''with open('someFile.csv', 'w') as file:
        write = csv.writer(file, dialect = Dialect.delimiter)
        write.writerow('e')'''
    
    file = open('RegLogTheta')
    txt = file.read()
    p = re.compile("[' '+]")
    print(re.findall(p,txt))
    print(re.sub("[\d[ +]\d]", " & ", txt))
    
        