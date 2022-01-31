#!/usr/bin/python3

import numpy as np
import pylab as pl
    
data = np.loadtxt("test_learning_fcl_filters.dat")
#
pl.title('Full run')
# compound
pl.subplot(411)
pl.plot(data[:,0])
pl.ylabel('input')
pl.ylim([0,1.5])
#

pl.subplot(412)
pl.plot(data[:,1])
pl.ylabel('err')
pl.ylim([0,1.5])
#

pl.subplot(413)
pl.plot(data[:,8])
pl.ylabel('output')
#

pl.subplot(414)
for i in range(10):
    pl.plot(data[:,int(9+i)])
pl.ylabel('filters')
#

#
pl.show()
