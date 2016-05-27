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
    
    x = pylab.linspace(0,30, num = 1000)
    x1 = (x - 5.0269)/(22.203 - 5.0269)
    
    y = 5.67002243 + 2.301*x1
    plt.plot(x,y)

    
    plt.show()