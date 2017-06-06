#ifndef __Layer_H_
#define __Layer_H_

#include "neuron.h"

class Layer {
	
public:

	int nNeurons;
	int nInputs;
	Neuron** neurons;

	Layer(int _nNeurons, int _nInputs);
	~Layer();

	void calcOutputs();
	void doLearning();

	// sets the global ICO error for the input layer
	void setError(float _error);

	// sets the error for the output layer
	void setError(int i, float _error);

	// retrieves the error
	float getError(int i);

	// this is used to copy the output from the previous
	// layer into this input layer or to the sensor inputs
	void setInput(int inputIndex, float input);

	void setLearningRate(float _learningRate);

	void initWeights(float _max);

	float getOutput(int index) {
		return neurons[index]->getOutput();
	}
	
};

#endif
