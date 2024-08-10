#ifndef __FEEDFORWARD_CLOSEDLOOP_LEARNING_H_
#define __FEEDFORWARD_CLOSEDLOOP_LEARNING_H_

#include "fcl/globals.h"
#include "fcl/layer.h"
#include "fcl/neuron.h"
#include "fcl/bandpass.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <vector>

/** Main class of Feedforward Closed Loop Learning.
 * Create an instance of this class to do the
 * learning. It will create the whole network
 * with an input layer,  layers and an
 * output layer. Learning is done iterative
 * by first setting the input values and errors
 * and then calling doStep().
 *
 * (C) 2017,2018-2022, Bernd Porr <bernd@glasgowneuro.tech>
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
	 * \param num_of_neurons_per_layer_array Number of neurons in each layer
	 * \param _num_layers Number of  layer (needs to match with array above)
	 **/
	FeedforwardClosedloopLearning(
		const int num_input, 
		const std::vector<int> &num_of_neurons_per_layer
	);

	/** Destructor
         * De-allocated any memory
         **/
	~FeedforwardClosedloopLearning();

	/** Performs the simulation step
         * \param input Array with the input values
         * \param error Array of the error signals
         **/
	void doStep(const std::vector<double> &input, const std::vector<double> &error);

	/** Gets the output from one of the output neurons
         * \param index: The index number of the output neuron.
         * \return The output value of the output neuron.
         **/
	double getOutput(int index) {
		return layers[n_neurons_per_layer.size()-1]->getOutput(index);
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
         **/
	void setActivationFunction(FCLNeuron::ActivationFunction _activationFunction);

	/** Inits the weights in all layers
         * \param max Maximum value of the weights.
         * \param initBias If the bias also should be initialised.
         * \param weightInitMethod See Neuron::WeightInitMethod for the options.
         **/
	void initWeights(double max = 0.001,
			 int initBias = 1,
			 FCLNeuron::WeightInitMethod weightInitMethod = FCLNeuron::MAX_OUTPUT_RANDOM);

	/** Seeds the random number generator
         * \param s An arbitratry number.
         **/
	void seedRandom(int s) { srand(s); };

	/** Sets globally the bias
         * \param _bias Sets globally the bias input to all neurons.
         **/
	void setBias(double _bias);

	/** Gets the total number of layers
         * \return The total number of all layers.
         **/
	int getNumLayers() {return (int)n_neurons_per_layer.size();};

	/** Gets a pointer to a layer
         * \param i Index of the layer.
         * \return A pointer to a layer class.
         **/
	FCLLayer* getLayer(unsigned i) {assert (i<=n_neurons_per_layer.size()); return layers[i];};

	/** Gets the output layer
         * \return A pointer to the output layer which is also a Layer class.
         **/
	FCLLayer* getOutputLayer() {return layers[n_neurons_per_layer.size()-1];};

	/** Gets the number of inputs
	 * \return The number of inputs
	 **/
	int getNumInputs() {return ni;}

	/** Returns all Layers
         * \return Returns a two dimensional array of all layers.
         **/
	FCLLayer** getLayers() {return layers;};

	/** Saves the whole network
         * \param name: filename
         **/
	bool saveModel(const char* name);

	/** Loads the while network
         * \param name: filename
         **/
	bool loadModel(const char* name);

	

private:
	unsigned ni;
	std::vector<int> n_neurons_per_layer;

	double learningRateDiscountFactor = 1;

	long int step = 0;

	double errorPrev = 0;

	double errorPrevPrev = 0;

	FCLLayer** layers;

	// should be called to relay layer index to the layer
	void setDebugInfo();

	void doLearning();
	void setStep();

};
#endif
