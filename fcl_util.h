#ifndef __FEEDFORWARD_CLOSEDLOOP_LEARNING_UTIL_H_
#define __FEEDFORWARD_CLOSEDLOOP_LEARNING_UTIL_H_

#include "fcl/globals.h"
#include "fcl/layer.h"
#include "fcl/neuron.h"
#include "fcl/bandpass.h"
#include "fcl.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/**
 * Derived classes of the FeedforwardClosedloopLearning class
 * for special functionality
 **/


class FeedforwardClosedloopLearningWithFilterbank : public FeedforwardClosedloopLearning {
	/**
	 * FeedforwardClosedloopLearning with Filterbank at each input
	 **/
public:
	/** Constructor: FCL with a filter bank at the input
	 * Every input feeds internally into a has a filter bank of num_filtersInput 
	 * filters. This allows for a temporal distribution of the inputs.
	 * \param num_of_inputs Number of inputs in the input layer
         * \param num_of_neurons_per_layer_array Number of neurons in each layer
	 * \param _num_layers Number of  layer (needs to match with array above)
         * \param num_filtersInput Number of filters at the input layer, 0 = no filterbank
         * \param num_filters Number of filters in the hiddel layers (usually zero)
         * \param _minT Minimum/first temporal duration of the 1st filter
         * \param _maxT Maximum/last temporal duration of the last filter
         **/
	FeedforwardClosedloopLearningWithFilterbank(
			int num_of_inputs,
			int* num_of_neurons_per_layer_array,
			int num_layers,
			int num_filtersInput,
			double minT,
			double maxT);

	/**
	 * Destructor
	 **/
	~FeedforwardClosedloopLearningWithFilterbank();

	/** Performs the simulation step
         * \param input Array with the input values
         * \param error Array of the error signals
         **/
	void doStep(double* input, double* error);

	/** Python wrapper function. Not public.
         **/
	void doStep(double* input, int n1, double* error, int n2);

	double getFilterOutput(int inputIdx, int filterIdx) {
		const int idx = inputIdx * nFiltersPerInput + filterIdx;
		assert((idx >= 0) || (idx < (nFiltersPerInput * nInputs)));
		return filterbankOutputs[idx];
	}

	int getNFiltersPerInput() {
		return nFiltersPerInput;
	}

private:
	const double dampingCoeff = 0.51;
	Bandpass ***bandpass = 0;
	double* filterbankOutputs = 0;
	int nFiltersPerInput = 0;
	int nInputs = 0;
};

#endif
