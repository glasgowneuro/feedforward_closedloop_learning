#!/usr/bin/python3

import numpy as np
import pylab as pl
import sys
#
if len(sys.argv) < 2:
    print("Need the .dat filename")
    exit(1)
    
data = np.loadtxt(sys.argv[1])
#
pl.title('Full run')
# compound
pl.subplot(311)
pl.plot(data[:,0])
pl.ylabel('input')
pl.ylim([0,1.5])
#

pl.subplot(312)
pl.plot(data[:,1])
pl.ylabel('err')
pl.ylim([0,1.5])
#

pl.subplot(313)
pl.plot(data[:,8])
pl.ylabel('output')
#

#
pl.show()
