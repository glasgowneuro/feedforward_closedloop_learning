#ifndef __FEEDFORWARD_CLOSEDLOOP_LEARNING_H_
#define __FEEDFORWARD_CLOSEDLOOP_LEARNING_H_

#include "fcl/globals.h"
#include "fcl/layer.h"
#include "fcl/neuron.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>


/** Main class of Feedforward Closed Loop Learning.
 * Create an instance of this class to do the
 * learning. It will create the whole network
 * with an input layer, hidden layers and an
 * output layer. Learning is done iterative
 * by first setting the input values and errors
 * and then calling doStep().
 *
 * (C) 2017,2018, Bernd Porr <bernd@glasgowneuro.tech>
 * (C) 2017,2018, Paul Miller <paul@glasgowneuro.tech>
 *
 * GNU GENERAL PUBLIC LICENSE
 * Version 3, 29 June 2007
 *
 **/
class FeedforwardClosedloopLearning {

public:
	
	/** Constructor: FCL without any filters
	 * \param num_of_inputs Number of inputs in the input layer
	 * \param num_of_hidden_neurons_per_layer_array Number of neurons in each layer
	 * \param _num_hid_layers Number of hidden layer (needs to match with array above)
	 * \param num_outputs Number of output in the output layer
	 **/
	FeedforwardClosedloopLearning(
			int num_of_inputs,
			int* num_of_hidden_neurons_per_layer_array,
			int _num_hid_layers,
			int num_outputs);

	/** Constructor: FCL with filters for both the input and hidden layers
	 * \param num_of_inputs Number of inputs in the input layer
         * \param num_of_hidden_neurons_per_layer_array Number of neurons in each layer
	 * \param _num_hid_layers Number of hidden layer (needs to match with array above)
	 * \param num_outputs Bumber of output in the output layer
         * \param num_filtersInput Number of filters at the input layer, 0 = no filterbank
         * \param num_filtersHidden Number of filters in the hiddel layers (usually zero)
         * \param _minT Minimum/first temporal duration of the 1st filter
         * \param _maxT Maximum/last temporal duration of the last filter
         **/
	FeedforwardClosedloopLearning(
			int num_of_inputs,
			int* num_of_hidden_neurons_per_layer_array,
			int _num_hid_layers,
			int num_outputs,
			int num_filtersInput,
			int num_filtersHidden,
			double _minT,
			double _maxT);

	/** Destructor
         * De-allocated any memory
         **/
	~FeedforwardClosedloopLearning();

	/** Performs the simulation step
         * \param input Array with the input values
         * \param error Array of the error signals
         **/
	void doStep(double* input, double* error);

	/** Python wrapper function. Not public.
         **/
	void doStep(double* input, int n1, double* error, int n2);

	/** Gets the output from one of the output neurons
         * \param index: The index number of the output neuron.
         * \return The output value of the output neuron.
         **/
	double getOutput(int index) {
		return layers[num_hid_layers]->getOutput(index);
	}

	/** Sets globally the learning rate
         * \param learningRate Sets the learning rate for all layers and neurons.
         **/
	void setLearningRate(double learningRate);

	/** Sets how the learnign rate increases or decreases from layer to layer
         * \param _learningRateDiscountFactor A factor of >1 means higher learning rate in deeper layers.
         **/
	void setLearningRateDiscountFactor(double _learningRateDiscountFactor) {
		learningRateDiscountFactor = _learningRateDiscountFactor;
	}

	/** Sets a typical weight decay scaled with the learning rate
         * \param decay The larger the faster the decay.
         **/
	void setDecay(double decay);

	/** Sets the global momentum for all layers
         * \param momentum Defines the intertia of the weight change over time.
         **/
	void setMomentum(double momentum);

	/** Sets the activation function of the Neuron
         * \param _activationFunction: See Neuron::ActivationFunction for the different options.
         **/
	void setActivationFunction(Neuron::ActivationFunction _activationFunction);

	/** Inits the weights in all layers
         * \param max Maximum value of the weights.
         * \param initBias If the bias also should be initialised.
         * \param weightInitMethod See Neuron::WeightInitMethod for the options.
         **/
	void initWeights(double max = 0.001,
			 int initBias = 1,
			 Neuron::WeightInitMethod weightInitMethod = Neuron::MAX_OUTPUT_RANDOM);

	/** Seeds the random number generator
         * \param s An arbitratry number.
         **/
	void seedRandom(int s) { srand(s); };

	/** Sets globally the bias
         * \param _bias Sets globally the bias input to all neurons.
         **/
	void setBias(double _bias);

	/** Returns the number of hidden layers
         * \return Integer value of how many hidden layers exist (total number is +1).
         **/
	int getNumHidLayers() {return num_hid_layers;};

	/** Gets the total number of layers
         * \return The total number of all layers.
         **/
	int getNumLayers() {return num_hid_layers+1;};

	/** Gets a pointer to a layer
         * \param i Index of the layer.
         * \return A pointer to a layer class.
         **/
	Layer* getLayer(int i) {assert (i<=num_hid_layers); return layers[i];};

	/** Gets the output layer
         * \return A pointer to the output layer which is also a Layer class.
         **/
	Layer* getOutputLayer() {return layers[num_hid_layers];};

	/** Returns all Layers
         * \return Returns a two dimensional array of all layers.
         **/
	Layer** getLayers() {return layers;};

	/** Sets if the learning should be the derivative of the error in each neuron
         * \param useIt: If one the derivative of the error signal is used for learning.
         **/
	void setUseDerivative(int useIt);

	/** Saves the whole network
         * \param name: filename
         **/
	bool saveModel(const char* name);

	/** Loads the while network
         * \param name: filename
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
