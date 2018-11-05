#ifndef __FEEDBACK_CLOSED_LOOP_LEARNING_H_
#define __FEEDBACK_CLOSED_LOOP_LEARNING_H_

/**
 * GNU GENERAL PUBLIC LICENSE
 * Version 3, 29 June 2007
 *
 * (C) 2017,2018, Bernd Porr <bernd@glasgowneuro.tech>
 * (C) 2017,2018, Paul Miller <paul@glasgowneuro.tech>
 **/

#include "fcl/globals.h"
#include "fcl/layer.h"
#include "fcl/neuron.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>


//#define DEBUG_DFL

class FeedbackClosedloopLearning {

public:
	// deep fbl without any filters
	FeedbackClosedloopLearning(
			int num_of_inputs,
			int* num_of_hidden_neurons_per_layer_array,
			int _num_hid_layers,
			int num_outputs);

	// deep fbl with filters for both the input and hidden layer
	// filter number >0 means: filterbank
	// filter number = 0 means layer without filters
	// filter parameters: are in time steps. For ex, minT = 10 means
	// a response of 10 time steps for the first filter and that goes
	// up to maxT time steps, for example maxT = 100 or so.
	FeedbackClosedloopLearning(
			int num_of_inputs,
			int* num_of_hidden_neurons_per_layer_array,
			int _num_hid_layers,
			int num_outputs,
			int num_filtersInput,
			int num_filtersHidden,
			double _minT,
			double _maxT);

	// destructor
	~FeedbackClosedloopLearning();

	// here is where all the magic is happening
	void doStep(double* input, double* error);

	// here is where all the magic is happening with array range checks
	void doStep(double* input, int n1, double* error, int n2);

	// get the output of the network
	double getOutput(int index) {
		return layers[num_hid_layers]->getOutput(index);
	}

	// set globally the learning rate
	void setLearningRate(double learningRate);

	// sets how the learnign rate increases or decreases in deeper layers
	void setLearningRateDiscountFactor(double _learningRateDiscountFactor) {
		learningRateDiscountFactor = _learningRateDiscountFactor;
	}

	// global momentum for all layers
	void setMomentum(double momentum);

	void setActivationFunction(Neuron::ActivationFunction _activationFunction);

	void initWeights(double max = 0.001,
			 int initBias = 1,
			 Neuron::WeightInitMethod weightInitMethod = Neuron::MAX_OUTPUT_RANDOM);

	void seedRandom(int s) { srand(s); };

	void setBias(double _bias);

	void setDecay(double decay);
	
	int getNumHidLayers() {return num_hid_layers;};
	int getNumLayers() {return num_hid_layers+1;};
	
	Layer* getLayer(int i) {assert (i<=num_hid_layers); return layers[i];};
	Layer* getOutputLayer() {return layers[num_hid_layers];};
	Layer** getLayers() {return layers;};

	void setUseDerivative(int useIt);

	bool saveModel(const char* name);
	bool loadModel(const char* name);

	

private:

	int ni;
	int nh;
	int no;
	int* n_hidden;
	int num_hid_layers;

	int nfInput;
	int nfHidden;
	double minT,maxT;

	double learningRateDiscountFactor = 1;

	long int step = 0;

	Layer** layers;

	// should be called to relay layer index to the layer
	void setDebugInfo();

	void doLearning();
	void setStep();

};

#endif
