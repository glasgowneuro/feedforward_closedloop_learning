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
		neurons[i]->setError(_error);
	}
}

void Layer::setLearningRate(float _learningRate) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setLearningRate(_learningRate);
	}
}

void Layer::initWeights(float _max) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->initWeights(_max);
	}
}

void Layer::setError(int i, float _error) {
	neurons[i]->setError(_error);
}

float Layer::getError(int i) {
	return neurons[i]->getError();
}

// setting a single input to all neurons
void Layer::setInput(int inputIndex, float input) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setInput(inputIndex,input);
	}
}
