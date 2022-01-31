#!/usr/bin/python3


import feedforward_closedloop_learning as fcl
import numpy as np
import matplotlib.pyplot as plt


print("testFCLWithFilters")

with open('test_fcl_filt_py.csv', 'wb') as csvfile:
    csvfile.close()
    
with open('test_fcl_filt_py.csv', 'ab') as csvfile:
    # two input neurons, two hidden ones and one output neuron
    # two filters and min temp filter is 10 pixels and max 100 pixels
    nFiltersInput = 5
    minT = 3
    maxT = 15
    net = fcl.FeedforwardClosedloopLearningWithFilterbank(2, [2,1], nFiltersInput, minT,maxT)
    # init the weights
    net.initWeights(0.001,0,fcl.Neuron.MAX_OUTPUT_CONST)
    net.setBias(0)
    net.setLearningRate(0.0001)
    #net.random_seed(10)
    # create the input arrays in numpy fashion
    inp = 0
    err = 0

    maxstep = 10000
    outp = np.zeros(maxstep)
    a = np.zeros(maxstep)
    b = np.zeros(maxstep)
    rep = 200

    for i in range(maxstep):
        if (((i%rep) > 100) and ((i%rep)<105)):
            inp = 1
        else :
            inp = 0
        if (((i%rep) > 105) and ((i%rep)<110) and (i < 9000)):
            err= 1
        else :
            err= 0
        # print(inp,err)
        net.doStep([inp,0],[err,err])
        # gets the output of the output neuron
        outp[i] = net.getOutput(0)
        a[i]=net.getLayer(0).getNeuron(0).getWeight(0)
        b[i]=net.getLayer(1).getNeuron(0).getWeight(0)
        np.savetxt(csvfile,np.hstack((inp,err,outp[i])),delimiter="\t",newline="\t")
        crlf="\n"
        csvfile.write(crlf.encode())

plt.figure(1)
plt.title("output")
plt.plot(outp)

plt.figure(2)
plt.title("Layer 0 weight")
plt.plot(a)

plt.figure(3)
plt.title("Layer 1 weight")
plt.plot(b)

plt.show()
