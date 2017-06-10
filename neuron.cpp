#include "neuron.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

Neuron::Neuron(int _nInputs, int _nFilters, double _minT, double _maxT) {
	nInputs = _nInputs;
	nFilters = _nFilters;
	minT = _minT;
	maxT = _maxT;
	weights = new double*[nInputs];
	if (nFilters>0) {
		bandpass = new Iir::Butterworth::LowPass<IIRORDER>**[nInputs];
	} else {
		bandpass = NULL;
		nFilters = 1;
	}
	for(int i=0;i<nInputs;i++) {
		weights[i] = new double[nFilters];
		if (bandpass != NULL) {
			bandpass[i] = new Iir::Butterworth::LowPass<IIRORDER>*[nFilters];
			double fs = 1;
			double fmin = fs/maxT;
			double fmax = fs/minT;
			double df = (fmax-fmin)/((double)nFilters);
			double f = fmin;
#ifdef DEBUG
			fprintf(stderr,"fmin=%f,fmax=%f,df=%f\n",fmin,fmax,df);
#endif
			for(int j=0;j<_nFilters;j++) {
				bandpass[i][j] = new Iir::Butterworth::LowPass<IIRORDER>;
#ifdef DEBUG
				fprintf(stderr,"bandpass[%d][%d]->setup(2,%f,%f,%f)\n",
					i,j,fs,f,df);
#endif
				bandpass[i][j]->setup(2,fs,f);
				f = f + df;
				for(int k=0;k<(maxT*10);k++) {
					float a = 0;
					if (k==minT) {
						a = 1;
					}
					float b = bandpass[i][j]->filter(a);
					assert(b != NAN);
					assert(b != INFINITY);
				}
			}
		}
	}
	inputs = new double[nInputs];
	prevInputs = new double[nInputs];
	sum = 0;
	output = 0;
	bias = 0;
	error = 0;
	learningRate = 0;
	for(int i=0;i<nInputs;i++) {
		for(int j=0;j<nFilters;j++) {
			weights[i][j] = 0;
		}
		inputs[i] = 0;
		prevInputs[i] = 0;
	}	
}

Neuron::~Neuron() {
	delete [] weights;
	delete [] inputs;
	delete [] prevInputs;
}


void Neuron::calcOutput() {
	sum = 0;
	for(int i=0;i<nInputs;i++) {
		for(int j = 0;j<nFilters;j++) {
			if (bandpass == NULL) {
				sum = sum + weights[i][j] * inputs[i];
			} else {
				sum = sum + weights[i][j] * bandpass[i][j]->filter(inputs[i]-prevInputs[i]);
				prevInputs[i] = inputs[i];
			}
		}
	}
#ifdef LINEAR_OUTPUT
	output = sum;
#else
	output = tanh(sum);
#endif
}


void Neuron::doLearning() {
	for(int i=0;i<nInputs;i++) {
		for(int j=0;j<nFilters;j++) {
			weights[i][j] = weights[i][j] + inputs[i] * error * learningRate;
		}
	}	
}


void Neuron::initWeights(double _max) {
	for(int i=0;i<nInputs;i++) {
		for(int j=0;j<nFilters;j++) {
			weights[i][j] = ((double)random())/((double)RAND_MAX)*_max;
		}
	}	
}
