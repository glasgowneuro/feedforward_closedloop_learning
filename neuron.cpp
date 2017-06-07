#include "neuron.h"
#include <math.h>
#include <stdlib.h>

Neuron::Neuron(int _nInputs) {
	nInputs = _nInputs;
	weights = new float[nInputs];
	inputs = new float[nInputs];
	sum = 0;
	output = 0;
	bias = 0;
	error = 0;
	learningRate = 0;
	for(int i=0;i<nInputs;i++) {
		weights[i] = 0;
		inputs[i] = 0;
	}	
}

Neuron::~Neuron() {
	delete [] weights;
	delete [] inputs;
}


void Neuron::calcOutput() {
	sum = 0;
	for(int i=0;i<nInputs;i++) {
		sum = sum + weights[i] * inputs[i];
	}
#ifdef LINEAR_OUTPUT
	output = sum;
#else
	output = tanh(sum);
#endif
}


void Neuron::doLearning() {
	for(int i=0;i<nInputs;i++) {
		weights[i] = weights[i] + inputs[i] * error * learningRate;
	}	
}


void Neuron::initWeights(float _max) {
	for(int i=0;i<nInputs;i++) {
		weights[i] = ((float)random())/((float)RAND_MAX)*_max;
	}	
}
