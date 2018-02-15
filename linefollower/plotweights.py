#!/usr/bin/python3

import sys
import threading
import math
import time

from matplotlib import pyplot as plt
import numpy as np

plt.ion()
plt.show()
ln = [False,False,False,False]


def plotOneMatrix(i):
    global ln
    if ln[i]:
        ln[i].remove()
    plt.title(str(i))
    w1 = np.loadtxt("layer"+str(i)+".dat");
    ln[i] = plt.imshow(w1,cmap='gray',interpolation='none')


while True:
    plt.subplot(221)
    plotOneMatrix(0)   
    plt.subplot(222)
    plotOneMatrix(1)   
    plt.subplot(223)
    plotOneMatrix(2)   
#    plt.subplot(224)
#    plotOneMatrix(3)   
    plt.draw()   
    plt.pause(10)     

