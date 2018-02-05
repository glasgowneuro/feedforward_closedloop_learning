import numpy as np

class Trace:

    def __init__(self, ntaps=5, tau=1):
        self.bufferFIR = None
        self.coeffFIR = None
        self.ntaps = ntaps
        self.tau = tau
        self.actualOutput = 0
        self.oldOutput = 0
        self.diff = 0.
        self.norm = 1.
    @staticmethod
    def calCoeffTrace(ntaps, tau):
        temp = np.array(range(ntaps))
        coeffFIR = np.exp(-temp*tau)

        return coeffFIR

    def filter(self, x):

        #shift
        self.bufferFIR = np.concatenate(([0.], self.bufferFIR), axis=0)[:self.ntaps]
        self.bufferFIR[0] = x
#        print ("buf: ", self.bufferFIR.shape, " coeff: ", self.coeffFIR.shape)
        output = np.dot(self.bufferFIR, self.coeffFIR) / self.norm
        self.actualOutput = output
        self.diff = self.actualOutput - self.oldOutput
        self.oldOutput = self.actualOutput

        return output













