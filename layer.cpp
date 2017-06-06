#include "layer.h"
#include "neuron.h"

Layer::Layer(int _nNeurons, int _nInputs) {
	nNeurons = _nNeurons;
	nInputs = _nInputs;

	neurons = new Neuron*[nNeurons];

	for(int i=0;i<nNeurons;i++) {
		neurons[i] = new Neuron(nInputs);
	}
}

Layer::~Layer() {
	for(int i=0;i<nNeurons;i++) {
		delete neurons[i];
	}
	delete [] neurons;
}

void Layer::calcOutputs() {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->calcOutput();
	}
}

void Layer::doLearning() {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->doLearning();
	}
}


void Layer::setError(float _error) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->error = _error;
	}
}

void Layer::setLearningRate(float _learningRate) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->learningRate = _learningRate;
	}
}

void Layer::initWeights(float _max) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->initWeights(_max);
	}
}

void Layer::setError(int i, float _error) {
	neurons[i]->error = _error;
}

float Layer::getError(int i) {
	return neurons[i]->error;
}

// setting a single input to all neurons
void Layer::setInput(int inputIndex, float input) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->inputs[inputIndex]=input;
	}
}
