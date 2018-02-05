from scipy import signal
import numpy as np
from trace import Trace

class IcoReflex:

    def __init__(self, num_filters=1, filter_type='IIR', freqResp="low"):

        num_inputs=1
        print ("construcing ICO: num_inputs: ", num_inputs, " num_filters: ", num_filters)
        self.n_inputs = num_inputs
        self.n_filters = num_filters
        self.ntaps = 5
        self.tau = 1
        self.filterBank = []

        # note that we seem to need an additional dimension for the inputs so that lfilter thinks it's a time series of one sample
        self.curr_input = np.zeros([num_inputs, 1])
        self.inputs = np.zeros([num_inputs, self.ntaps])
        self.lastInputs = np.zeros(num_inputs) # needed??
        self.filteredOutputs = np.zeros([num_filters, num_inputs])
        self.weights = np.zeros([num_filters, num_inputs])
        self.actualActivity = 0.
        self.diff = 0.
        self.oldOutput = 0.
        self.norm = 1.
        self.filterType = filter_type
        self.IIROrder = 2

        self.weights[0, 0] = 1  # reflex is always 1

        # put the filter (single filter only) for the reflex first
#        self.filterBank.append([Trace(ntaps=self.ntaps)])

        if self.filterType == 'trace':
            for i in range(num_filters):
                ntaps = int(float(self.ntaps) / float(i + 1))
                self.filterBank.append(Trace.calCoeffTrace(ntaps, self.tau))
            print (self.filterBank)

        elif self.filterType == 'IIR':
            maxFreq = 0.5
            minFreq = 0.5
            if (freqResp == 'low'):
                self.a = np.zeros([num_filters, self.IIROrder+1])
                self.b = np.zeros([num_filters, self.IIROrder+1])
                self.zf_old = np.zeros([num_filters, num_inputs, self.IIROrder])
            elif (freqResp == 'band'):
                self.a = np.zeros([num_filters, self.IIROrder*2 + 1])
                self.b = np.zeros([num_filters, self.IIROrder*2 + 1])
                self.zf_old = np.zeros([num_filters, num_inputs, self.IIROrder*2])

            for i in range(num_filters):
                if (num_filters < 2):
                    freq = minFreq
                else:
                    freq = maxFreq - float(i) * (maxFreq - minFreq)/(float(num_filters)-1)
                print ("Freq: ", freq)
                if (freqResp == 'low'):
                    self.b[i,:], self.a[i,:] = signal.butter(self.IIROrder, freq, analog=False) # 3rd order lowpass
                elif (freqResp == 'band'):
                    self.b[i,:], self.a[i,:] = signal.butter(self.IIROrder, [minFreq, maxFreq], 'band', analog=False)

                zi = signal.lfilter_zi(self.b[i,:], self.a[i,:]) # initialise state

                # set initial filter state
                temp = np.empty(num_inputs)
                temp.fill(0.0)
                self.zf_old[i,:,:] = np.outer(temp, zi)

            print ("a: ", self.a)
            print ("b: ", self.b)

        else:
            print ("unknown filter type, exiting")
            return

    def setCurrInput(self, input):
        self.curr_input[:,0] = input

    def filter(self):
        if (self.filterType == 'trace'):
            # shift - append the new column, then trim off the old one
            self.inputs = (np.c_[self.curr_input[:,0], self.inputs])[:,:self.ntaps]

            # dot product with filter coefficients
            self.diff = 0.
            for i in range(self.n_filters):
                self.filteredOutputs[i,:] = (self.inputs[:,:(self.filterBank[i]).shape[0]]).dot(self.filterBank[i]) / self.norm

            # derivative of the reflex
            self.diff = self.filteredOutputs[0,0] - self.oldOutput
            self.oldOutput = self.filteredOutputs[0,0]

        else:
            self.diff = 0.
#            self.filteredOutputs[0,0] = self.curr_input[0,0]

            for i in range(self.n_filters):
                a = self.a[i]
                b = self.b[i]
                zfold = self.zf_old[i,:,:]
#                print ("MIN: ", np.amin(self.curr_input[1:19201,0:1]), " MAX: ", np.amax(self.curr_input[1:19201,0:1]))

                # we have to do this in an awkward way because we can't assign an (n,1) array to a row in filtereredOutputs
                z1, zf = signal.lfilter(b, a, self.curr_input[0:,0:1], zi=zfold)
                self.filteredOutputs[i,0:] = np.ndarray.flatten(z1)
                # copy state to old_state
                self.zf_old[i,:,:] = zf


            # derivative of reflex
            self.diff = self.filteredOutputs[0, 0] - self.oldOutput
            self.oldOutput = self.filteredOutputs[0, 0]


    # that's the one to call
    def prediction(self, curr_step, inputs):
#        print ("PREDICT: ", curr_step)
        self.setCurrInput(inputs)
        self.filter()

        self.actualActivity = (np.ndarray.flatten(self.filteredOutputs)).dot(np.ndarray.flatten(self.weights))


        return self.actualActivity


    # at the moment, only support setting the entire input block in one operation.
    def setInput(self, f):
        self.inputs = f


    def saveInputs(self, curr_step):
        print("saving input images...")
        for i in range(self.n_filters):
            np.save('/tmp/icoSteer-' + str(i) + "-" + str(curr_step), self.filteredOutputs[i,:])
