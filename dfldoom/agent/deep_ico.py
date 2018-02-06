"""
Pure Python implementation of a Back-Propagation Neural Network using the
hyperbolic tangent as the sigmoid squashing function.

Original Author: Neil Schemenauer <nas@arctrix.com>
Modified Author: James Howard <james.w.howard@gmail.com>
Modified Author: Bernd Porr <mail@berndporr.me.uk>

GNU public license
==

Modified by Jerzy Karczmarczuk <jerzy.karczmarczuk@unicaen.fr>
* Converted to Python 3 (syntax)
* Uses numpy


"""

import csv
import numpy as np


# Make a matrix (NumPy to speed this up)
def makeZ(I, J):
    return np.zeros((I,J),dtype='double')

    
# calculate a random number where:  a <= rand < b
def rnd(a, b, shp):
    return (b-a)*np.random.rand(*shp)+a


class NN(object):
    def __init__(self, num_input, num_hidden, num_output, 
                 learnig_rate=0.005, momentum=0.1, init_weight = 0.01,
                 derivative = True,
                 do_tan = True):
        """NN constructor.
        
        ni, nh, no are the number of input, hidden and output nodes.
        regression is used to determine if the Neural network will be trained 
        and used as a classifier or for function regression.
        """
        
        self.N = learnig_rate
        self.M = momentum
        self.de = derivative
        self.dt = do_tan
        
        #Number of input, hidden and output nodes.
        self.ni = num_input
        self.nh = num_hidden
        self.no = num_output

        # activations for nodes
        self.ai = np.ones(self.ni)
        self.ah = np.ones(self.nh)
        self.ao = np.ones(self.no)
        
        # create weights
        # set them to random values
        self.wi=rnd(-init_weight,init_weight,(self.ni,self.nh))
        self.wo=rnd(-init_weight,init_weight,(self.nh,self.no))
        # last change in weights for momentum   
        self.ci = makeZ(self.ni, self.nh)
        self.co = makeZ(self.nh, self.no)


    # derivative of our sigmoid function, in terms of the output (i.e. y)
    def dsigmoid(self, y):
        if (self.de) :
            return 1.0 - y**2
        else :
            return y


    def sigmoid(self, x):
        if (self.dt) :
            return np.tanh(x)
        else :
            return x


    def step(self, inputs, errors):
        if len(inputs) != self.ni:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni):
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            total = 0.0
            for i in range(self.ni):
                total += self.ai[i] * self.wi[i,j]
            self.ah[j] = self.sigmoid(total)

        # output activations
        for k in range(self.no):
            total = 0.0
            for j in range(self.nh):
                total += self.ah[j] * self.wo[j,k]
            self.ao[k] = total
            self.ao[k] = self.sigmoid(total)

        self.learn(errors);
	
        return np.copy(self.ao)  # self.ao[:]



    def learn(self, errors):
        if len(errors) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for hidden
        hidden_deltas = np.zeros(self.nh)
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error += errors[k]*self.wo[j,k]
            hidden_deltas[j] = self.dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = errors[k]*self.ah[j]
                self.wo[j,k] = self.wo[j,k] + self.N*change + self.M*self.co[j,k]
                self.co[j,k] = change

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*self.ai[i]
                self.wi[i,j] = self.wi[i,j] + self.N*change + self.M*self.ci[i,j]
                self.ci[i,j] = change



    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            outp = print(self.wo[j])
            print(outp)

def demoDEEPICO():
    with open('test.csv', 'wb') as csvfile:
        csvfile.close()
        
    with open('test.csv', 'ab') as csvfile:
        net = NN(2, 2, 1, momentum = 0, derivative = True, do_tan = True)
        inp = np.zeros(2)
        err = np.zeros(1)
        for i in range(100):
            if (i > 10) :
                inp[0] = 1
            if ((i > 20) and (i<90)) :
                err[0] = 1
            else :
                err[0] = 0
            output = net.step(inp,err.T)
            print(output)
            np.savetxt(csvfile,np.hstack((inp,err,output)),delimiter="\t",newline="\t")
            crlf="\n"
            csvfile.write(crlf.encode())

if __name__ == '__main__':
    demoDEEPICO()
