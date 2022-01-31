#include "fcl_util.h"
#include <math.h>

/**
 * GNU GENERAL PUBLIC LICENSE
 * Version 3, 29 June 2007
 *
 * (C) 2017, Bernd Porr <bernd@glasgowneuro.tech>
 * (C) 2017, Paul Miller <paul@glasgowneuro.tech>
 **/

FeedforwardClosedloopLearningWithFilterbank::FeedforwardClosedloopLearningWithFilterbank(
			int num_of_inputs,
			int* num_of_neurons_per_layer_array,
			int num_layers,
			int num_filtersInput,
			double minT,
			double maxT) : FeedforwardClosedloopLearning(
				num_of_inputs * num_filtersInput,
				num_of_neurons_per_layer_array,
				num_layers) {
#ifdef DEBUG
	fprintf(stderr,"Creating instance of FeedforwardClosedloopLearningWithFilterbank.\n");
#endif	
	nFiltersPerInput = num_filtersInput;
	nInputs = num_of_inputs;
	assert((nInputs*nFiltersPerInput) == getNumInputs());
	bandpass = new Bandpass**[num_of_inputs];
	filterbankOutputs = new double[num_of_inputs * num_filtersInput];
	for(int i=0;i<num_of_inputs;i++) {
		bandpass[i] = new Bandpass*[num_filtersInput];
		double fs = 1;
		double fmin = fs/maxT;
		double fmax = fs/minT;
		double df = (fmax-fmin)/((double)(num_filtersInput-1));
		double f = fmin;
#ifdef DEBUG
		fprintf(stderr,"bandpass: fmin=%f,fmax=%f,df=%f\n",fmin,fmax,df);
#endif
		for(int j=0;j<num_filtersInput;j++) {
			bandpass[i][j] = new Bandpass();
#ifdef DEBUG
			fprintf(stderr,"bandpass[%d][%d]->setParameters(%f,%f)\n",
				i,j,f,dampingCoeff);
#endif
			bandpass[i][j]->setParameters(f,dampingCoeff);
			f = f + df;
#ifdef DEBUG
			for(int k=0;k<maxT;k++) {
				double a = 0;
				if (k==minT) {
					a = 1;
				}
				double b = bandpass[i][j]->filter(a);
				assert(b != NAN);
				assert(b != INFINITY);
			}
#endif
			bandpass[i][j]->reset();
		}
	}
}

FeedforwardClosedloopLearningWithFilterbank::~FeedforwardClosedloopLearningWithFilterbank() {
	delete[] filterbankOutputs;
	for(int i=0;i<nInputs;i++) {
		for(int j=0;j<nFiltersPerInput;j++) {
			delete bandpass[i][j];
		}
		delete[] bandpass[i];
	}
	delete[] bandpass;
}


void FeedforwardClosedloopLearningWithFilterbank::doStep(double* input, double* error) {
	for(int i=0;i<nInputs;i++) {
		for(int j=0;j<nFiltersPerInput;j++) {
			filterbankOutputs[i*nFiltersPerInput+j] = bandpass[i][j]->filter(input[i]);	
		}
	}
	FeedforwardClosedloopLearning::doStep(filterbankOutputs,error);
}


void FeedforwardClosedloopLearningWithFilterbank::doStep(double* input, int n1, double* error, int n2) {
#ifdef DEBUG
	fprintf(stderr,"doStep: n1=%d,n2=%d\n",n1,n2);
#endif
	if (n1 != nInputs) {
		fprintf(stderr,"Input array dim mismatch: got: %d, want: %d\n",n1,nInputs);
		return;
	}
	if (n2 != getLayer(0)->getNneurons()) {
		fprintf(stderr,
			"Error array dim mismatch: got: %d, want: %d "
			"which is the number of neurons in the 1st hidden layer!\n",
			n2,getLayer(0)->getNneurons());
		return;
	}
	FeedforwardClosedloopLearningWithFilterbank::doStep(input,error);
}
