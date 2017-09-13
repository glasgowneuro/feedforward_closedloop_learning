#ifndef __Deep_FEEDBACK_LEARNING_H_
#define __Deep_FEEDBACK_LEARNING_H_

/**
 * GNU GENERAL PUBLIC LICENSE
 * Version 3, 29 June 2007
 *
 * (C) 2017, Bernd Porr <mail@berndporr.me.uk>, <bernd@glasgowneuro.tech>
 * (C) 2017, Paul Miller <nlholdem@hotmail.com>
 **/

#include "globals.h"
#include "layer.h"
#include "neuron.h"
#include <stdio.h>
#include <stdlib.h>



// do the proper derivative of the activation function
// #define DO_DERIV_ACTIVATION

//#define DEBUG_DFL

class DeepFeedbackLearning {

public:
	// deep ico without any filters
	DeepFeedbackLearning(
			int num_of_inputs,
			int* num_of_hidden_neurons_per_layer_array,
			int _num_hid_layers,
			int num_outputs);

	// deep ico with filters for both the input and hidden layer
	// filter number >0 means: filterbank
	// filter number = 0 means layer without filters
	// filter parameters: are in time steps. For ex, minT = 10 means
	// a response of 10 time steps for the first filter and that goes
	// up to maxT time steps, for example maxT = 100 or so.
	DeepFeedbackLearning(
			int num_of_inputs,
			int* num_of_hidden_neurons_per_layer_array,
			int _num_hid_layers,
			int num_outputs,
			int num_filtersInput,
			int num_filtersHidden,
			double _minT, double _maxT);
	
	~DeepFeedbackLearning();

	enum Algorithm { backprop = 0, ico = 1 };

	void doStep(double* input, double* error);

	void doStep(double* input, int n1, double* error, int n2);

	double getOutput(int index) {
		return layers[num_hid_layers]->getOutput(index);
	}

	void setLearningRate(double learningRate);

	void setMomentum(double momentum);

	void setAlgorithm(Algorithm _algorithm) { algorithm = _algorithm; }

	Algorithm getAlgorithm() { return algorithm; }

	void initWeights(double max = 0.001, int initBias = 1, Neuron::WeightInitMethod weightInitMethod = Neuron::MAX_OUTPUT_RANDOM);

	void seedRandom(int s) { srandom(s); };

	void setBias(double _bias);
	
	int getNumHidLayers() {return num_hid_layers;};
	Layer* getLayer(int i) {assert (i<=num_hid_layers); return layers[i];};
	Layer* getOutputLayer() {return layers[num_hid_layers];};
	Layer** getLayers() {return layers;};

	void setUseDerivative(int useIt);

private:

	int ni;
	int nh;
	int no;
	int* n_hidden;
	int num_hid_layers;

	int nfInput;
	int nfHidden;
	double minT,maxT;

	long int step = 0;

	Layer** layers;

	Algorithm algorithm;

	// should be called to relay layer index to the layer
	void setDebugInfo();

	void doStepBackprop(double* input, double* error);
	void doStepForwardprop(double* input, double* error);

	void doLearning();
	void setStep();

#ifdef DO_DERIV_ACTIVATION
	double dsigm(double y) { return (1.0 - y*y); };
#else
	double dsigm(double y) { return 1.0; };
#endif
};

#endif
