'''
Created on Jun 1, 2016

@author: ahanagrawal
'''
import numpy as np
import ExamScoreVisual as es
import matplotlib.pyplot as plt
import scipy.optimize as sci
import re


theta = np.zeros(shape = (7,4))
i = 0
f = open("RegLogTheta", 'r')
x = f.readline().rstrip()
while x != '':
    q = x.split()
    for j in range(len(q)):
        theta[i/4][i % 4] = q[j]
        i += 1
    x = f.readline().rstrip()


def thetaFunc(y, theta, x):

    deg = 6
    
    spot = 0
    sum = 0
    for i in range(1, deg + 1):
        for j in range(i + 1):
            sum += theta[spot % 7][spot / 7] * x**(i - j) * y**(j)
            spot += 1
    return sum


if __name__ == "__main__":
    data = np.loadtxt("ex2/ex2data2.txt", delimiter = ",")
    Admits, Rejects = es.segAdmits(data)
    
    plt.scatter(Admits[:,0], Admits[:,1], marker = "+")
    plt.scatter(Rejects[:,0], Rejects[:,1], c = "y", marker = "o")
    plt.xlabel('Test II Scores', fontsize = 12)
    plt.ylabel('Test I Scores', fontsize = 12)
    
#     pattern = re.compile(b'[\d]')
#     theta = np.fromregex("RegLogTheta", regexp = r"\s+,(\d+)\s+,", dtype = [(np.float128)])
#     theta = getTheta()

#     txt = open("RegLogTheta", "r").read()
#     txt = re.sub(",", " ", txt)
#     txt = txt.split()
#     
#     txt2 = open("RegLogTheta2", "r").read()
#     txt2 = re.sub(",", " ", txt2)
#     txt2 = txt2.split()
    'txt = re.sub("[\s+]", ",", txt)'
#     txt = txt.split(",")
#     txt = "RegLogTheta".replace(b"\n", b"")
#     theta = np.array(txt, float)
#     theta2 = np.array(txt2,float)
    
    x = np.linspace(-1, 1.5, 100)
    y = np.linspace(-1,1.5,100)
    z = np.empty((100,100))
    
    for i in range(len(x)):
        for j in range(len(y)):
            z[i][j] = thetaFunc(y[j], theta, x[i])
    
    plt.contour(x,y,z, levels = [0])
    
#     plt.figure(2)
#     
#     for i in range(len(x)):
#         for j in range(len(y)):
#             z[i][j] = thetaFunc(y[j], theta2, x[i])
#     
#     plt.contour(x,y,z, levels = [0])
#     
    plt.show()

    
    