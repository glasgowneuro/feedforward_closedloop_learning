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
	const int num_of_inputs,
	const std::vector<int> &num_of_neurons_per_layer,
	const int num_filtersInput,
	const double minT,
	const double maxT) : FeedforwardClosedloopLearning(
		num_of_inputs * num_filtersInput,
		num_of_neurons_per_layer) {
#ifdef DEBUG
	fprintf(stderr,"Creating instance of FeedforwardClosedloopLearningWithFilterbank.\n");
#endif	
	nFiltersPerInput = num_filtersInput;
	nInputs = num_of_inputs;
	assert((nInputs*nFiltersPerInput) == getNumInputs());
	bandpass = new FCLBandpass**[num_of_inputs];
	filterbankOutputs.resize(num_of_inputs * num_filtersInput);
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

FeedforwardClosedloopLearningWithFilterbank::~FeedforwardClosedloopLearningWithFilterbank() {
	for(int i=0;i<nInputs;i++) {
		for(int j=0;j<nFiltersPerInput;j++) {
			delete bandpass[i][j];
		}
		delete[] bandpass[i];
	}
	delete[] bandpass;
}


void FeedforwardClosedloopLearningWithFilterbank::doStep(const std::vector<double> &input,
							 const double err) {
	if (input.size() != (unsigned)nInputs) {
		char tmp[256];
		sprintf(tmp,"Input array dim mismatch: got: %ld, want: %d.",input.size(),nInputs);
		#ifdef DEBUG
		fprintf(stderr,"%s\n",tmp);
		#endif
		throw tmp;
	}
	for(int i=0;i<nInputs;i++) {
		for(int j=0;j<nFiltersPerInput;j++) {
			filterbankOutputs[i*nFiltersPerInput+j] = bandpass[i][j]->filter(input[i]);
		}
	}
	FeedforwardClosedloopLearning::doStep(filterbankOutputs, err);
}
