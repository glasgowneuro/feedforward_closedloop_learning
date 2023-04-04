#include "cldl_filterbank.h"
#include <math.h>

/**
 * GNU GENERAL PUBLIC LICENSE
 * Version 3, 29 June 2007
 *
 * (C) 2017, Bernd Porr <bernd@glasgowneuro.tech>
 * (C) 2017, Paul Miller <paul@glasgowneuro.tech>
 **/

ClosedloopDeepLearningWithFilterbank::ClosedloopDeepLearningWithFilterbank(
			const int num_of_inputs,
			const int* num_of_neurons_per_layer_array,
			const int num_layers,
			const int num_filtersInput,
			const double minT,
			const double maxT) : CLDLNet(
				num_layers,
				num_of_neurons_per_layer_array,
				num_of_inputs * num_filtersInput
				) {
#ifdef DEBUG
	fprintf(stderr,"Creating instance of FeedforwardClosedloopLearningWithFilterbank.\n");
#endif	
	nFiltersPerInput = num_filtersInput;
	nInputs = num_of_inputs;
	assert((nInputs*nFiltersPerInput) == getnInputs());
	bandpass = new FCLBandpass**[num_of_inputs];
	filterbankOutputs = new double[num_of_inputs * num_filtersInput];
	for(int i=0;i<num_of_inputs;i++) {
		bandpass[i] = new FCLBandpass*[num_filtersInput];
		double fs = 1;
		double fmin = fs/maxT;
		double fmax = fs/minT;
		double df = (fmax-fmin)/((double)(num_filtersInput-1));
		double f = fmin;
#ifdef DEBUG
		fprintf(stderr,"bandpass: fmin=%f,fmax=%f,df=%f\n",fmin,fmax,df);
#endif
		for(int j=0;j<num_filtersInput;j++) {
			bandpass[i][j] = new FCLBandpass();
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

ClosedloopDeepLearningWithFilterbank::~ClosedloopDeepLearningWithFilterbank() {
	delete[] filterbankOutputs;
	for(int i=0;i<nInputs;i++) {
		for(int j=0;j<nFiltersPerInput;j++) {
			delete bandpass[i][j];
		}
		delete[] bandpass[i];
	}
	delete[] bandpass;
}


void ClosedloopDeepLearningWithFilterbank::doStep(const double* input, const double* errors) {
	for(int i=0;i<nInputs;i++) {
		for(int j=0;j<nFiltersPerInput;j++) {
			filterbankOutputs[i*nFiltersPerInput+j] = bandpass[i][j]->filter(input[i]);
		}
	}
	CLDLNet::doStep(filterbankOutputs,errors);
}
