#include "deep_ico.h"
#include <math.h>

Deep_ICO::Deep_ICO(int num_input, int num_hidden, int num_output) {

	ni = num_input;
	nh = num_hidden;
	no = num_output;
	
	hiddenLayer = new Layer(nh,ni);
	outputLayer = new Layer(no,nh);

	setLearningRate(0.001);

}

Deep_ICO::~Deep_ICO() {
	delete hiddenLayer;
	delete outputLayer;
}

void Deep_ICO::doStep(double* input, double* error) {

	for(int i=0;i<ni;i++) {
		hiddenLayer->setInput(i,input[i]);
	}
	for(int i=0;i<no;i++) {
		outputLayer->setError(i,error[i]);
	}
	// let's first calc all activities
	hiddenLayer->calcOutputs();
	// now that we have the outputs from the hidden layer
	// we can shovel them into the next layer
	for(int i=0;i<nh;i++) {
		// get the output of a neuron in the input layer
		float v = hiddenLayer->getNeuron(i)->getOutput();
		// set that output as an input to the next layer which
		// is distributed to all neurons
		outputLayer->setInput(i,v);
	}
	// now let's calc the output which can then be sent out
	outputLayer->calcOutputs();

	// Calcualte the errors for the hidden layer
	for(int i=0;i<hiddenLayer->getNneurons();i++) {
		float err = 0;
		for(int j=0;j<outputLayer->getNneurons();j++) {
			err = err + outputLayer->getNeuron(j)->getWeight(i) *
				outputLayer->getNeuron(j)->getError();
		}
		hiddenLayer->getNeuron(i)->setError(dsigm(hiddenLayer->getNeuron(i)->getOutput()) * err);
	}


	outputLayer->doLearning();
	hiddenLayer->doLearning();
}

void Deep_ICO::setLearningRate(float rate) {
	hiddenLayer->setLearningRate(rate);
	outputLayer->setLearningRate(rate);
}


void Deep_ICO::initWeights(float max) {
	hiddenLayer->initWeights(max);
	outputLayer->initWeights(max);
}
