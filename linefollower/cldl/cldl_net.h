#pragma once

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <ctgmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <iostream>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <numeric>
#include <vector>

#include "cldl_layer.h"

/** Net is the main class used to set up a neural network used for
 * closed-loop Deep Learning. It initialises all the layers and the
 * neurons internally.
 *
 * (C) 2019,2020-2023, Bernd Porr <bernd@glasgowneuro.tech>
 * (C) 2019,2020, Sama Daryanavard <2089166d@student.gla.ac.uk>
 *
 * GNU GENERAL PUBLIC LICENSE
 **/
class CLDLNet {

public:

/** Constructor: The neural network that performs the learning.
 * \param _nLayers Total number of hidden layers, excluding the input layer
 * \param _nNeurons A pointer to an int array with number of
 * neurons for all layers need to have the length of _nLayers.
 * \param _nInputs Number of Inputs to the network
 **/
	CLDLNet(const int _nLayers,
	    const int * const _nNeurons,
	    const int _nInputs);

/**
 * Destructor
 * De-allocated any memory
 **/
	virtual ~CLDLNet();

/**
 * @brief does a realtime step
 * 
 */
virtual void doStep(const double* input, const double* error);

/** Dictates the initialisation of the weights and biases
 * and determines the activation function of the neurons.
 * \param _wim weights initialisation method,
 * see Neuron::weightInitMethod for different options
 * \param _bim biases initialisation method,
 * see Neuron::biasInitMethod for different options
 * \param _am activation method,
 * see Neuron::actMethod for different options
 **/
	void initNetwork(CLDLNeuron::weightInitMethod _wim, CLDLNeuron::biasInitMethod _bim, CLDLNeuron::actMethod _am);
	
/** Sets the learning rate.
 * \param _learningRate Sets the learning rate for
 * all layers and neurons.
 **/
	void setLearningRate(double _w_learningRate, double _b_learningRate = 0.0);
	
/** Sets the inputs to the network in each iteration
 * of learning, needs to be placed in an infinite loop.
 * @param _inputs A pointer to the array of inputs
 */
	void setInputs(const double *_inputs);
/**
 * It propagates the inputs forward through the network.
 */
	void propInputs();

	/**
	 * Propagates the error backward
	 **/
	void propErrorBackward();

/**
 * Sets the error at the output layer to be propagated backward.
 * @param _leadError The closed-loop error for learning
 */
	void setErrors(const double* _leadError);

/**
 * It provides a measure of how the magnitude of the error changes through the layers
 * to alarm for vanishing or exploding gradients.
 * \param _whichError choose what error to monitor, for more information see Neuron::whichError
 * \param _whichGradient choose what gradient of the chosen error to monitor,
 * for more information see Layer::whichGradient
 * @return Returns the ratio of the chosen gradient in the last layer to the the first layer
 */
	double getGradient(CLDLLayer::whichGradient _whichGradient);

/**
 * Requests that all layers perform one iteration of learning
 */
	void updateWeights();

/**
 * Allows Net to access each layer
 * @param _layerIndex the index of the chosen layer
 * @return A pointer to the chosen Layer
 */
	CLDLLayer *getLayer(int _layerIndex);
/**
 * Allows the user to access the activation output of a specific neuron in the output layer only
 * @param _neuronIndex The index of the chosen neuron
 * @return The value at the output of the chosen neuron
 */
	double getOutput(int _neuronIndex);
/**
 * Allows the user to access the weighted sum output of a specific neuron in output layer only
 * @param _neuronIndex The index of the chosen neuron
 * @return The value at the sum output of the chosen neuron
 */
	double getSumOutput(int _neuronIndex);

/**
 * Informs on the total number of hidden layers (excluding the input layer)
 * @return Total number of hidden layers in the network
 */
	int getnLayers();
/**
 * Informs on the total number of inputs to the network
 * @return Total number of inputs
 */
	int getnInputs();

/**
 * Allows for monitoring the overall weight change of the network.
 * @return returns the Euclidean wight distance of all neurons in the network from their initial value
 */
	double getWeightDistance();

/**
 * Allows for monitoring the weight change in a specific layer of the network.
 * @param _layerIndex The index of the chosen layer
 * @return returns the Euclidean wight distance of neurons in the chosen layer from their initial value
 */
	double getLayerWeightDistance(int _layerIndex);
/**
 * Grants access to a specific weight in the network
 * @param _layerIndex Index of the layer that contains the chosen weight
 * @param _neuronIndex Index of the neuron in the chosen layer that contains the chosen weight
 * @param _weightIndex Index of the input to which the chosen weight is assigned
 * @return returns the value of the chosen weight
 */
	double getWeights(int _layerIndex, int _neuronIndex, int _weightIndex);

/**
 * Informs on the total number of neurons in the network
 * @return The total number of neurons
 */
	int getnNeurons();

/**
 * Saves the temporal changes of all weights in all neurons into files
 */
	void saveWeights();
/**
 * Snaps the final distribution of all weights in a specific layer,
 * this is overwritten every time the function is called
 */
	void snapWeights(string prefix);
	void snapWeightsMatrixFormat(string prefix);
/**
 * Prints on the console a full tree of the network with the values of all weights and outputs for all neurons
 */
	void printNetwork();

private:

	/**
	 * Total number of hidden layers
	 */
	int nLayers = 0;

    /**
	 * total number of neurons
	 */
	int nNeurons = 0;

        /**
	 * total number of weights
	 */
	int nWeights = 0;
  
	/**
	 * total number of inputs
	 */
	int nInputs = 0;
  
	/**
	 * total number of outputs
	 */
	int nOutputs = 0;
  
	/**
	 * A double pointer to the layers in the network
	 */
	CLDLLayer **layers = 0;
  
	/**
	 * A pointer to the inputs of the network
	 */
	const double *inputs = 0;
  
	/**
	 * A pointer to the gradient of the error
	 */
	double *errorGradient = NULL;
};
