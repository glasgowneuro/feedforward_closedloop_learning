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


using namespace std;

/**
 * This is the class for creating neurons inside the Layer class.
 * This is the building block class of the network.
 */

class CLDLNeuron {
public:

	/**
	 * Constructor for the Neuron class: it initialises a neuron with specific number fo inputs to that neuron
	 * @param _nInputs
	 */
	CLDLNeuron(int _nInputs);
	/**
	 * Destructor
	 * De-allocated any memory
	 */
	~CLDLNeuron();

	/**
	 * Options for method of initialising biases
	 * 0 for initialising all weights to zero
	 * 1 for initialising all weights to one
	 * 2 for initialising all weights to a random value between 0 and 1
	 */
	enum biasInitMethod { B_NONE = 0, B_RANDOM = 1 };
	/**
	 * Options for method of initialising weights
	 * 0 for initialising all weights to zero
	 * 1 for initialising all weights to one
	 * 2 for initialising all weights to a random value between 0 and 1
	 */
	enum weightInitMethod { W_ZEROS = 0, W_ONES = 1, W_RANDOM = 2, W_ONES_NORM = 3, W_RANDOM_NORM = 5,  };
	/**
	 * Options for activation functions of the neuron
	 * 0 for using the logistic function
	 * 1 for using the hyperbolic tan function
	 * 2 for unity function (no activation)
	 */
	enum actMethod {Act_Sigmoid = 1, Act_Tanh = 2, Act_ReLU = 3, Act_NONE = 0};
	/**
	 * Options for choosing an error to monitor the gradient of
	 * 0 for monitoring the error that propagates backward
	 * 1 for monitoring the error that propagates from the middle and bilaterally
	 * 2 for monitoring the error that propagates forward
	 */
	enum whichError {onBackwardError = 0, onMidError = 1, onForwardError = 2};

	/**
	 * Initialises the neuron with the given methods for weight/bias initialisation and for activation function.
	 * It also specifies the index of the neuron and the index of the layer that contains this neuron.
	 * @param _neuronIndex The index of this neuron
	 * @param _layerIndex The index of the layer that contains this neuron
	 * @param _wim The method of initialising the weights, refer to weightInitMethod for more information
	 * @param _bim The method of initialising the biases, refer to biasInitMethod for more information
	 * @param _am The function used for activation of neurons, refer to actMethod for more information
	 */
	void initNeuron(int _neuronIndex, int _layerIndex, weightInitMethod _wim, biasInitMethod _bim, actMethod _am);

	/** Sets the learning rate.
	 * @param _learningRate Sets the learning rate for this neuron.
	 **/
	void setLearningRate(double _learningRate, double _b_learningRate);

	/**
	 * Sets the inputs to this neuron that is located in the first hidden layer
	 * @param _index Index of the input
	 * @param _value Value of the input
	 */
	void setInput(int _index, double _value);

	/**
	 * Sets the inputs to this neuron that can be located in any layer other than the first hidden layer
	 * @param _index index of the input
	 * @param _value value of the input
	 */
	void propInputs(int _index, double _value);

	/**
	 * Calculates the output of the neuron by performing a weighed sum of all inputs to this neuron and activating the sum
	 * @return Returns a boolean to report whether or not this neuron has exploding output
	 */
	void calcOutput();

	/**
	 * Sets the error of the neuron in the first hidden layer that is to be propagated forward
	 * @param _value value of the error
	 */
	void setError(double _value);

	/**
	 * Allows accessing the error of this neuron
	 * @return the value of the error
	 */
	double getError();

	/**
	 * Performs one iteration of learning, that is: it updates all the weights assigned to each input to this neuron
	 */
	void updateWeights();
	
	/**
	 * Performs the activation of the sum output of the neuron
	 * @param _sum the weighted sum of all inputs
	 * @return activation of the sum
	 */
	inline double doActivation(const double sum) const {
		switch(actMet){
		case Act_Sigmoid:
			return (1/(1+(exp(-sum)))) - 0.5;
		case Act_Tanh:
			return tanh(sum);
		case Act_ReLU:
			if (sum > 0) return sum; else return 0;
		case Act_NONE:
			return sum;
		}
		return sum;
	}

	/**
	 * Performs inverse activation on any input that is passed to this function
	 * @param _input the input value
	 * @return the inverse activation of the input
	 */
	inline double doActivationPrime(const double input) const {
		switch(actMet){
		case Act_Sigmoid:
			return 1 * (0.5 + doActivation(input)) * (0.5 - doActivation(input));
		case Act_Tanh:
			return 1 - pow (tanh(input), 2);
		case Act_ReLU:
			if (sum > 0) return 1; else return 0;
		case Act_NONE:
			return 1;
		}
		return 1;
	}

	/**
	 * Sets the internal backprop error
	 * @param _input the input value
	 */
	inline void setBackpropError(const double upstreamDeltaErrorSum) {
		error = doActivationPrime(getSumOutput()) * upstreamDeltaErrorSum;
	}

	/**
	 * Requests the output of this neuron
	 * @return the output of the neuron after activation
	 */
	inline double getOutput() {
		return output;
	}

	/**
	 * Requests the sum output of the neuron
	 * @return returns the sum output of the neuron before activaiton
	 */
	inline double getSumOutput() {
		return sum;
	}

	/**
	 * Requests a specific weight
	 * @param _inputIndex index of the input to which the chosen weight is assigned
	 * @return Returns the chosen weight
	 */
	double getWeights(int _inputIndex);

	/**
	 * Requests a inital value of a specific weight
	 * @param _inputIndex index of the input to which the weight is assigned
	 * @return teh inital value of the weight
	 */
	double getInitWeights(int _inputIndex);

	/**
	 * Requests for overall change of all weights contained in this neuron
	 * @return the overal weight change
	 */
	double getWeightChange();

	/**
	 * Requests for the maximum weights located in this neuron
	 * @return Returns the max weight
	 */
	double getMaxWeight(){
		return maxWeight;
	}
	
	/**
	 * Requests for the minimum weights located in this neuron
	 * @return Returns the min weight
	 */
	double getMinWeight(){
		return minWeight;
	}

	/**
	 * Requests for the total sum of weights located in this neuron
	 * @return Returns the sum of weights
	 */
	double getSumWeight(){
		return weightSum;
	}

	/**
	 * Requests the weight distance of all weighs in this neuron
	 * @return returns the sqr of the total weight change in this neuron
	 */
	double getWeightDistance();
	
	/**
	 * Requests the total number of inputs to this neuron
	 * @return total number of inputs
	 */
	int getnInputs();

	/**
	 * Saves the temporal weight change of all weights in this neuron into a file
	 */
	void saveWeights();

	/**
	 * Prints on the console a full description of all weights, inputs and outputs for this neuron
	 */
	void printNeuron();

	/**
	 * Sets the weights of the neuron
	 * @param _index index of the weight
	 * @param _weight value of the weight
	 */
	inline void setWeight(int _index, double _weight) {
		assert((_index >= 0) && (_index < nInputs));
		weights[_index] = _weight;
	}

private:
	// initialisation:
	int nInputs = 0;
	int myLayerIndex = 0;
	int myNeuronIndex = 0;
	double *initialWeights = 0;

	double w_learningRate = 0;
	double b_learningRate = 0;
    
	int iHaveReported = 0;

	double bias = 0;
	double sum = 0;
	double output = 0;

	//learning:
	double *weights = 0;
	double *inputs = 0;
	double weightSum = 0;
	double maxWeight = 1;
	double minWeight = 1;
	double weightChange=0;
	double weightsDifference = 0;
	int actMet = 0;

	double error = 0;
};
