#include "deep_ico.h"
#include <math.h>

Deep_ICO::Deep_ICO(int _nInputsPerNeuron, int _neuronsPerLayer) {

	neuronsPerLayer = _neuronsPerLayer;
	nInputsPerNeuron = _nInputsPerNeuron;

	hiddenLayer = new Layer(neuronsPerLayer,nInputsPerNeuron);
	outputLayer = new Layer(neuronsPerLayer,nInputsPerNeuron);
	
}

Deep_ICO::~Deep_ICO() {
	delete hiddenLayer;
	delete outputLayer;
}

void Deep_ICO::doStep() {
	// let's first calc all activities
	hiddenLayer->calcOutputs();
	// now that we have the outputs from the hidden layer
	// we can shovel them into the next layer
	for(int i=0;i<neuronsPerLayer;i++) {
		// get the output of a neuron in the input layer
		float v = hiddenLayer->neurons[i]->output;
		// set that output as an input to the next layer which
		// is distributed to all neurons
		outputLayer->setInput(i,v);
	}
	// now let's calc the output which can then be sent out
	outputLayer->calcOutputs();

	// Calcualte the errors for the hidden layer
	for(int i=0;i<hiddenLayer->nNeurons;i++) {
		float err = 0;
		for(int j=0;j<outputLayer->nInputs;j++) {
			err = err + outputLayer->neurons[i]->weights[j] * outputLayer->neurons[j]->error;
		}
		hiddenLayer->neurons[i]->error = err;
	}

	outputLayer->doLearning();
	hiddenLayer->doLearning();
}


void Deep_ICO::setLearningRate(float rate) {
	for(int i=0;i<outputLayer->nNeurons;i++) {
		hiddenLayer->setLearningRate(rate);
		outputLayer->setLearningRate(rate);
	}
}
