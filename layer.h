#ifndef __Layer_H_
#define __Layer_H_

#include "neuron.h"

class Layer {
	
public:

	Layer(int _nNeurons, int _nInputs);
	~Layer();

	void calcOutputs();
	void doLearning();

	// sets the global error for all neurons
	void setError(double _error);

	// sets the error individually
	void setError(int i, double _error);

	// sets all errors from an input array
	void setErrors(double *_errors);

	// retrieves the error
	double getError(int i);

	// this is used to copy the output from the previous
	// layer into this input layer or to the sensor inputs
	void setInput(int inputIndex, double input);

	// sets all inputs from an input array
	void setInputs(double *_inputs);

	// sets the learning rate of all neurons
	void setLearningRate(double _learningRate);

	// inits weights with a random value between -_max and max
	void initWeights(double _max);

	// gets the outpuut of one neuron
	inline double getOutput(int index) {
		return neurons[index]->getOutput();
	}

	// gets a pointer to one neuron
	inline Neuron* getNeuron(int index) {
		return neurons[index];
	}

	// number of neurons
	int getNneurons() { return nNeurons;}

	// number of inputs
	int getNinputs() { return nInputs;}

private:
	int nNeurons;
	int nInputs;
	Neuron** neurons;

};

#endif
