#ifndef __FEEDFORWARD_CLOSEDLOOP_LEARNING_UTIL_H_
#define __FEEDFORWARD_CLOSEDLOOP_LEARNING_UTIL_H_

#include "cldl_globals.h"
#include "cldl_layer.h"
#include "cldl_neuron.h"
#include "cldl_net.h"
#include "cldl_bandpass.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/**
 * Derived classes of the FeedforwardClosedloopLearning class
 * for special functionality
 **/


class ClosedloopDeepLearningWithFilterbank : public CLDLNet {
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
	ClosedloopDeepLearningWithFilterbank (
			int num_of_inputs,
			int* num_of_neurons_per_layer_array,
			int num_layers,
			int num_filtersInput,
			double minT,
			double maxT);

	/**
	 * Destructor
	 **/
	virtual ~ClosedloopDeepLearningWithFilterbank();

	/** Performs the simulation step
         * \param input Array with the input values
         * \param error Array of the error signals
         **/
	virtual void doStep(const double* input, const double* error) override;

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
	FCLBandpass ***bandpass = 0;
	double* filterbankOutputs = 0;
	int nFiltersPerInput = 0;
	int nInputs = 0;
};

#endif
