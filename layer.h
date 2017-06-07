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
	void setError(float _error);

	// sets the error individually
	void setError(int i, float _error);

	// retrieves the error
	float getError(int i);

	// this is used to copy the output from the previous
	// layer into this input layer or to the sensor inputs
	void setInput(int inputIndex, float input);

	// sets the learning rate of all neurons
	void setLearningRate(float _learningRate);

	// inits weights with a random value between -_max and max
	void initWeights(float _max);

	// gets the outpuut of one neuron
	inline float getOutput(int index) {
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
