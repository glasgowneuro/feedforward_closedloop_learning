#include "layer.h"
#include "neuron.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

Layer::Layer(int _nNeurons, int _nInputs, int _nFilters, double _minT, double _maxT) {
	nNeurons = _nNeurons;
	nInputs = _nInputs;
	nFilters = _nFilters;
	minT = _minT;
	maxT = _maxT;

	neurons = new Neuron*[nNeurons];

	for(int i=0;i<nNeurons;i++) {
		neurons[i] = new Neuron(nInputs,nFilters,minT,maxT);
	}

	initWeights(0);
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


void Layer::setError(double _error) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setError(_error);
	}
}

void Layer::setErrors(double* _errors) {
	for(int i=0;i<nNeurons;i++) {
		if (isnan(_errors[i])) {
			fprintf(stderr,"Layer::setErrors: _errors[%d]=%f\n",i,_errors[i]);
				exit(EXIT_FAILURE);
		}
		neurons[i]->setError(_errors[i]);
	}
}

void Layer::setLearningRate(double _learningRate) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setLearningRate(_learningRate);
	}
}

void Layer::setUseDerivative(int _useIt) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setUseDerivative(_useIt);
	}
}

void Layer::initWeights(double _max) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->initWeights(_max);
	}
}

void Layer::setError(int i, double _error) {
	neurons[i]->setError(_error);
}

double Layer::getError(int i) {
	return neurons[i]->getError();
}

// setting a single input to all neurons
void Layer::setInput(int inputIndex, double input) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setInput(inputIndex,input);
	}
}

// setting a single input to all neurons
void Layer::setInputs(double* inputs) {
	for(int j=0;j<nInputs;j++) {
		for(int i=0;i<nNeurons;i++) {
			neurons[i]->setInput(j,inputs[j]);
		}
	}
}
