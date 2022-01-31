#!/usr/bin/python3

import numpy as np
import pylab as pl
    
data = np.loadtxt("test_learning_fcl_filters.dat")
data2 = np.loadtxt("test_learning_fcl_filters2.dat")

#
pl.title('Full run')


pl.subplot(411)
pl.plot(data[:,0])
pl.ylabel('input')
pl.ylim([0,1.5])

pl.subplot(412)
for i in range(10):
    pl.plot(data2[:,i])
pl.ylabel('filters')

pl.subplot(413)
pl.plot(data[:,1])
pl.ylabel('err')
pl.ylim([0,1.5])


pl.subplot(414)
pl.plot(data[:,8])
pl.ylabel('output')

pl.show()
