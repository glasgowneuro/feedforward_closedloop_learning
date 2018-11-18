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
	
	/** 
	 * Constructor: FCL without any filters
	 * @num_of_inputs: number of inputs in the input layer
	 * @num_of_hidden_neurons_per_layer_array: number of neurons in each layer
	 * @_num_hid_layers: number of hidden layer (needs to match with array above)
	 * @num_outputs: number of output in the output layer
	 **/
	FeedbackClosedloopLearning(
			int num_of_inputs,
			int* num_of_hidden_neurons_per_layer_array,
			int _num_hid_layers,
			int num_outputs);

	/**
	 * Constructor: FCL with filters for both the input and hidden layers
	 * filter number >0 means: filterbank
	 * filter number = 0 means layer without filters
	 * @num_of_inputs: number of inputs in the input layer
         * @num_of_hidden_neurons_per_layer_array: number of neurons in each layer
	 * @_num_hid_layers: number of hidden layer (needs to match with array above)
	 * @num_outputs: number of output in the output layer
         * @num_filtersInput: number of filters at the input layer
         * @num_filtersHidden: number of filters in the hiddel layers (usually zero)
         * @_minT: minimum/first temporal duration of the 1st filter
         * @_maxT: maximum/last temporal duration of the last filter
         **/
	FeedbackClosedloopLearning(
			int num_of_inputs,
			int* num_of_hidden_neurons_per_layer_array,
			int _num_hid_layers,
			int num_outputs,
			int num_filtersInput,
			int num_filtersHidden,
			double _minT,
			double _maxT);

	/**
         * Destructor
         **/
	~FeedbackClosedloopLearning();

	/**
         * Performs the simulation step
         * @input: Array with the input values
         * @error: Array of the error signals
         **/
	void doStep(double* input, double* error);

	// For the python wrapper
	void doStep(double* input, int n1, double* error, int n2);

	/**
         * Gets the output from one of the output neurons
         **/
	double getOutput(int index) {
		return layers[num_hid_layers]->getOutput(index);
	}

	/**
         * Sets globally the learning rate
         * @learningRate for all layers and neurons.
         **/
	void setLearningRate(double learningRate);

	/**
         * Sets how the learnign rate increases or decreases from layer to layer
         * @_learningRateDiscountFactor: >1 means higher learning rate in deeper layers
         **/
	void setLearningRateDiscountFactor(double _learningRateDiscountFactor) {
		learningRateDiscountFactor = _learningRateDiscountFactor;
	}

	/**
         * Sets a typical weight decay scaled with the learning rate
         * @decay: >0, the larger the faster the decay
         **/
	void setDecay(double decay);

	/**
	 * Sets the global momentum for all layers
         **/
	void setMomentum(double momentum);

	/**
         * Sets the activation function of the Neuron
         * @_activationFunction: see Neuron::ActivationFunction for the different options
         **/
	void setActivationFunction(Neuron::ActivationFunction _activationFunction);

	/**
         * Inits the weights in all layers
         * @max: Maximum value of the weights
         * @initBias: If the bias also should be initialised
         * @weightInitMethod: see Neuron::WeightInitMethod for the options
         **/
	void initWeights(double max = 0.001,
			 int initBias = 1,
			 Neuron::WeightInitMethod weightInitMethod = Neuron::MAX_OUTPUT_RANDOM);

	/**
         * Seeds the random number generator
         **/
	void seedRandom(int s) { srand(s); };

	/**
         * Sets globally the bias
         **/
	void setBias(double _bias);

	/**
         * Returns the number of hidden layers
         **/
	int getNumHidLayers() {return num_hid_layers;};

	/**
         * Gets the total number of layers
         **/
	int getNumLayers() {return num_hid_layers+1;};

	/**
         * Gets a pointer to a layer
         * @i: Index of the layer
         **/
	Layer* getLayer(int i) {assert (i<=num_hid_layers); return layers[i];};

	/**
         * Gets the output layer
         **/
	Layer* getOutputLayer() {return layers[num_hid_layers];};

	/**
         * Returns all Layers
         **/
	Layer** getLayers() {return layers;};

	/**
         * Sets if the learning should be the derivative of the error in each neuron
         * @useIt: If one the derivative is used.
         **/
	void setUseDerivative(int useIt);

	/**
         * Saves the whole network
         * @name: filename
         **/
	bool saveModel(const char* name);

	/**
         * Loads the while network
         * @name: filename
         **/
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
