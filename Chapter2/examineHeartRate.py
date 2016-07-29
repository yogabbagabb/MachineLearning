'''
Created on Jun 19, 2016

@author: ahanagrawal
'''

import numpy as np
import matplotlib.pyplot as plt

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

if __name__ == '__main__':
    length = file_len("heartRate.txt")
    width = 8
    
    txt = open("heartRate.txt",'r')
    
    data = np.zeros((length,width), dtype = object)
    txt.readline()
    txt.readline()
    line = txt.readline()
    i = 0
    while (line != ""):
        dataRow = line.replace(",", " ").split()
#         data.append(dataRow)
        dataLen = min(len(dataRow),8)
        data[i,0:dataLen] = dataRow[0:dataLen]
        line = txt.readline()
        i += 1
        
#     print(data[0][1], data[0][2], data[0][3])
    colorMarkers = np.copy(data[:,4]).tolist()
    for i in range(len(colorMarkers)):
        if (colorMarkers[i] == '0'):
            colorMarkers[i] = 'g'
        else:
            colorMarkers[i] = 'r'
#1
    print(colorMarkers)
    print(data[:,0])
    print(data[:,1])
    plt.figure()
    plt.scatter(data[:,0], data[:,1], c = colorMarkers)
    plt.show()

        