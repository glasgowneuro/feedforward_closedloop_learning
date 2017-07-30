#!/usr/bin/python3

import deep_feedback_learning
import numpy as np
import matplotlib.pyplot as plt

print("testICOWithFilters and derivative")

with open('test_bp_filt_py.csv', 'wb') as csvfile:
    csvfile.close()
    
with open('test_bp_filt_py.csv', 'ab') as csvfile:
    # two input neurons, two hidden ones and one output neuron
    # two filters and min temp filter is 10 pixels and max 100 pixels
    nFiltersInput = 5
    nFiltersHidden = 5
    # nFiltersHidden = 0 means that the layer is linear without filters
    minT = 3
    maxT = 15
    nNeuronsHidden = 10
    nInputs = 2
    net = deep_feedback_learning.DeepFeedbackLearning(nInputs, [nNeuronsHidden,nNeuronsHidden], 1, nFiltersInput, nFiltersHidden, minT,maxT)
    # init the weights
    net.initWeights(0.001,0,deep_feedback_learning.Neuron.MAX_OUTPUT_CONST);
    net.setBias(0);
    net.setAlgorithm(deep_feedback_learning.DeepFeedbackLearning.ico);
    net.setLearningRate(0.001)
    net.getLayer(1).setLearningRate(1)
    net.getLayer(2).setLearningRate(1)
    net.setUseDerivative(1)
    #net.getLayer(1).setNormaliseWeights(1)
    #net.getLayer(2).setNormaliseWeights(1)
    #net.random_seed(10)
    # create the input arrays in numpy fashion
    inp = 0
    err = 0

    maxstep = 10000
    outp = np.zeros((maxstep,nNeuronsHidden))
    l0 = np.zeros((maxstep,nInputs))
    l1 = np.zeros((maxstep,nNeuronsHidden))
    l2 = np.zeros((maxstep,nNeuronsHidden))
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
        # does both forward propagation and backpropagation
        # print(inp,err)
        net.doStep([inp,0],np.linspace(err,err,nNeuronsHidden))
        # gets the output of the output neuron
        outp[i] = net.getOutput(0)
        for j in range(nInputs):
            l0[i,j]=net.getLayer(0).getNeuron(0).getWeight(j)
        for j in range(nNeuronsHidden):
            l1[i,j]=net.getLayer(1).getNeuron(0).getWeight(j)
            l2[i,j]=net.getLayer(2).getNeuron(0).getWeight(j)
        np.savetxt(csvfile,np.hstack((inp,err,outp[i])),delimiter="\t",newline="\t")
        crlf="\n"
        csvfile.write(crlf.encode())

plt.figure(1)
plt.plot(outp)

plt.figure(2)
plt.plot(l0)

plt.figure(3)
plt.plot(l1)

plt.figure(4)
plt.plot(l2)

plt.show()
