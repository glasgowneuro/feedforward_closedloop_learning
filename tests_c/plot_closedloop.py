import numpy as np
import pylab as pl
#
data = np.loadtxt('test_closedloop.dat');
#
fig = pl.figure(1)
#
a = 0
b = len(data[:,0])
pl.title('Full run');
# compound
pl.subplot(911);
pl.plot(data[a:b,0],data[a:b,1]);
pl.ylabel('pred');
# 
pl.subplot(912);
pl.plot(data[a:b,0],data[a:b,2]);
pl.ylabel('dist');
#
pl.subplot(913);
pl.plot(data[a:b,0],data[a:b,3]);
pl.ylabel('err');
#
pl.subplot(914);
pl.plot(data[a:b,0],data[a:b,4]);
pl.ylabel('v');
#
pl.subplot(915);
pl.plot(data[a:b,0],data[a:b,5]);
pl.ylabel('out');
#
pl.subplot(916);
pl.plot(data[a:b,0],data[a:b,6]);
pl.ylabel('weight');
#
pl.subplot(917);
pl.plot(data[a:b,0],data[a:b,7]);
pl.ylabel('weight');
#
pl.subplot(918);
pl.plot(data[a:b,0],data[a:b,8]);
pl.ylabel('weight');
#
pl.subplot(919);
pl.plot(data[a:b,0],data[a:b,9]);
pl.ylabel('weight');
pl.show();
