'''
Created on May 22, 2016

@author: ahanagrawal
'''

import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D
import pylab
import numpy as np
import HousePrices as hp

def main (*args):
    array = pylab.loadtxt("ex1/ex1data1.txt", dtype = float, delimiter = ",")
        
    ax1 = plt.subplot(111)
    ax1.scatter(array[:,0], array[:,1])
    
    
    x = pylab.linspace(0,30, num = 1000)
    x1 = (x - 5.0269)/(22.203 - 5.0269)
    
    y =  2.22412189 +  19.8519207*x1
    print(x[0:5],x1[0:5],y[0:5])
    plt.plot(x,y)


    '''fig2 = plt.figure(2)
    ax2 = fig2.gca()'''
#     plt.subplot(2,1,2)
#     plt.gca(projection = '3d')
#     
#     X,Y = hp.main()
#     plt.scatter(X[:,1], X[:, 2], Y)
    'plt.plot(X[:,0],Y)'
    plt.show()
if __name__ == "__main__":
    main()
    
    
