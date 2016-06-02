'''
Created on Jun 1, 2016

@author: ahanagrawal
'''
import numpy as np
import ExamScoreVisual as es
import matplotlib.pyplot as plt
import scipy.optimize as sci
import re

def getTheta():
    theta = np.zeros(29)
    i = 0
    f = open("RegLogTheta", 'r')
    x = f.readline().rstrip()
    while x != '':
        q = x.split("[\s,]")
        for j in range(len(q)):
            theta[i] = q[j]
            i += 1
        x = f.readline().rstrip()

def thetaFunc(y, theta, x):

    deg = 6
    
    spot = 0
    sum = 0
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
    
    x = np.linspace(-1, 1.5, 1000)
    '''pattern = re.compile(b'[\s,]')
    theta = np.fromregex("RegLogTheta", regexp = r"\s+,(\d+)\s+,", dtype = [(np.float128)])'''
#     theta = getTheta()

    txt = open("RegLogTheta", "r").read()
    txt = np.loadtxt(txt, delimiter = )
    
    
    y = np.zeros(1000)
    for i in range(len(x)):
        y[i] = sci.anderson(thetaFunc, xin = 0.5, args = (theta, x[i]))
    
    plt.plot(x,y)
    plt.show()

    
    