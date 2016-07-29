    '''
Created on Jun 1, 2016

@author: ahanagrawal
'''
import numpy as np
import ExamScoreVisual as es
import matplotlib.pyplot as plt
import scipy.optimize as sci
import RegLogRegr
import re


# theta = np.zeros(shape = (7,4))
# i = 0
# f = open("RegLogTheta", 'r')
# x = f.readline().rstrip()
# while x != '':
#     q = x.split()
#     for j in range(len(q)):
#         theta[i/4][i % 4] = q[j]
#         i += 1
#     x = f.readline().rstrip()


def thetaFunc(y, theta, x):

    deg = 6
    
    spot = 0
    sum = theta[spot]
    spot += 1
    for i in range(1, deg + 1):
        for j in range(i + 1):
            sum += theta[spot] * x**(i - j) * y**(j)
            spot += 1
    return sum


if __name__ == "__main__":
    data = np.loadtxt("ex2/ex2data2.txt", delimiter = ",")
    Admits, Rejects = es.segAdmits(data)
    
    plt.scatter(Admits[:,0], Admits[:,1], marker = "+")
    plt.scatter(Rejects[:,0], Rejects[:,1], c = "y", marker = "o")
    plt.xlabel('Test II Scores', fontsize = 12)
    plt.ylabel('Test I Scores', fontsize = 12)
    



    txt = open("RegLogTheta2", "r").read()
    txt = txt.replace(",", " ")
    txt = txt.split()
     
 
    theta = np.array(txt, float)

    
    x = np.linspace(-1, 1.5, 100)
    y = np.linspace(-1,1.5,100)
    xx, yy = np.meshgrid(x,y)
    z = np.empty((100,100))
    
    data = np.c_[xx.ravel(), yy.ravel()]
    data = RegLogRegr.constructVariations(np.c_[xx.ravel(), yy.ravel()], 6)
    oneArray = np.ones((len(data),1))
    data = np.hstack((oneArray, data))

    
    for i in range(len(x)):
        for j in range(len(y)):

            z[i][j] = thetaFunc(y[j], theta, x[i])
    z -= 1.2712452

    plt.contour(x,y,z)
    
    plt.figure(2)
    
    txt = open("RegLogTheta", "r").read()
    txt = txt.replace(",", " ")
    txt = txt.split()
     
 
    theta = np.array(txt, float)
    
    plt.scatter(Admits[:,0], Admits[:,1], marker = "+")
    plt.scatter(Rejects[:,0], Rejects[:,1], c = "y", marker = "o")
    
    xx,yy = np.meshgrid(x,y)
    for i in range(len(x)):
        for j in range(len(y)):
            z[i][j] = thetaFunc(yy[i][j], theta, xx[i][j])
    z -= 1.2712452
            

    plt.contour(xx,yy,z, levels = [0])
    
     
    plt.show()

    
    