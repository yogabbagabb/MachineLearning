'''
Created on May 22, 2016

@author: ahanagrawal
'''

import matplotlib.pyplot as plt
import pylab
import numpy as np




if __name__ == "__main__":
    
    array = pylab.loadtxt("ex1/ex1data1.txt", dtype = float, delimiter = ",")
    plt.scatter(array[:,0], array[:,1])
    
    x = pylab.linspace(0,10, 1000)
    
    
    y = 0.17024736 + 0.08690504*x
    plt.plot(x,y)
    
    plt.show()