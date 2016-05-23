'''
Created on May 22, 2016

@author: ahanagrawal
'''

import matplotlib.pyplot as plt
import pylab




if __name__ == "__main__":
    
    array = pylab.loadtxt("ex1/ex1data1.txt", dtype = float, delimiter = ",")
    plt.scatter(array[:,0], array[:,1])
    plt.show()