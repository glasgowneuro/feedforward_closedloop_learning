#include "layer.h"
#include "neuron.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

Layer::Layer(int _nNeurons, int _nInputs, int _nFilters, double _minT, double _maxT) {

	nNeurons = _nNeurons;
	nInputs = _nInputs;
	nFilters = _nFilters;
	minT = _minT;
	maxT = _maxT;
	normaliseWeights = 0;

	neurons = new Neuron*[nNeurons];

	for(int i=0;i<nNeurons;i++) {
		neurons[i] = new Neuron(nInputs,nFilters,minT,maxT);
	}

	initWeights(0,0,Neuron::CONST_WEIGHTS);
}

Layer::~Layer() {
	for(int i=0;i<nNeurons;i++) {
		delete neurons[i];
	}
	delete [] neurons;
}

void Layer::calcOutputs() {
	pthread_t t[nNeurons];
	for(int i=0;i<nNeurons;i++) {
		pthread_create(&t[i], NULL, neurons[i]->calcOutputThread, neurons[i]);
	}
	for(int i=0;i<nNeurons;i++) {
		pthread_join(t[i], NULL);
	}
}

void Layer::doLearning() {
	pthread_t t[nNeurons];
	if (maxDetLayer) {
		for(int i=0;i<nNeurons;i++) {
			pthread_create(&t[i], NULL, neurons[i]->doMaxDetThread, neurons[i]);
		}
	} else {
		for(int i=0;i<nNeurons;i++) {
			pthread_create(&t[i], NULL, neurons[i]->doLearningThread, neurons[i]);
		}
	}
	for(int i=0;i<nNeurons;i++) {
		pthread_join(t[i], NULL);
	}
	if (!normaliseWeights) return;
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->normaliseWeights();	
	}
}


void Layer::setNormaliseWeights(int _normaliseWeights) {
	normaliseWeights = _normaliseWeights;
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->normaliseWeights();
		neurons[i]->saveInitialWeights();
	}	
}


void Layer::setError(double _error) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setError(_error);
	}
}

void Layer::setErrors( double* _errors) {
	for(int i=0;i<nNeurons;i++) {
		if (isnan(_errors[i])) {
			fprintf(stderr,"Layer::%s L=%d, errors[%d]=%f\n",__func__,layerIndex,i,_errors[i]);
		}
		neurons[i]->setError(_errors[i]);
	}
}

void Layer::setBias(double  _bias) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setBias(_bias);
	}
}

void Layer::setLearningRate( double _learningRate) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setLearningRate(_learningRate);
	}
}

void Layer::setMomentum( double _momentum) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setMomentum(_momentum);
	}
}

void Layer::setUseDerivative( int _useIt) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setUseDerivative(_useIt);
	}
}

void Layer::initWeights( double max, int initBias, Neuron::WeightInitMethod weightInitMethod) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->initWeights(max,initBias,weightInitMethod);
	}
}

void Layer::setError( int i,  double _error) {
	neurons[i]->setError(_error);
}

double Layer::getError( int i) {
	return neurons[i]->getError();
}

// setting a single input to all neurons
void Layer::setInput(int inputIndex, double input) {
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setInput(inputIndex,input);
	}
}

// setting a single input to all neurons
void Layer::setDebugInfo(int _layerIndex) {
	layerIndex = _layerIndex;
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setDebugInfo(_layerIndex,i);
	}
}


// setting a single input to all neurons
void Layer::setStep(long int _step) {
	step = _step;
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setStep(step);
	}
}

double Layer::getWeightDistanceFromInitialWeights() {
	double distance = 0;
	for(int i=0;i<nNeurons;i++) {
		distance += neurons[i]->getWeightDistanceFromInitialWeights();
	}
	return distance;
}


	double weightDistanceFromInitialWeights();



void Layer::setInputs( double* inputs ) {
	double* inputp = inputs;
	inputp = inputs;
	for(int j=0;j<nInputs;j++) {
		Neuron** neuronsp = neurons;
		 double input = *inputp;
		inputp++;
		for(int i=0;i<nNeurons;i++) {
			(*neuronsp)->setInput(j,input);
			neuronsp++;
		}
	}
}


void Layer::setConvolution( int width,  int height) {
	float  d = round(sqrt(nNeurons));
	int dx = round(width/d);
	int dy = round(height/d);
	int mx = round(dx/2.0);
	int my = round(dy/2.0);
	for(int i=0;i<nNeurons;i++) {
		neurons[i]->setGeometry(width,height);
		neurons[i]->setMask(0);
		for(int x=0;x<dx;x++) {
			for(int y=0;y<dy;y++) {
				neurons[i]->setMask(x+mx-dx/2,y+my-dx/2,1);
			}
		}
		mx = mx + dx;
		if (mx > width) {
			mx = round(dx/2.0);
			my = my + dy;
		}
	}
}
