import numpy as np
import pylab as pl
#
data = np.loadtxt('test_deep_fbl_cpp_forward.dat')
#
pl.title('Activities in the network with random weights')
# compound
pl.subplot(411)
pl.plot(data[:,0])
pl.ylabel('input')
pl.ylim([0,0.2])
#

pl.subplot(412)
pl.plot(data[:,1])
pl.ylabel('hidden')
#

pl.subplot(413)
pl.plot(data[:,2])
pl.ylabel('hidden')

pl.subplot(414)
pl.plot(data[:,3])
pl.ylabel('output')
#

#
pl.show()
