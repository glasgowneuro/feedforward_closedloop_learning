#include "deep_ico.h"
#include <math.h>

Deep_ICO::Deep_ICO(int num_input, int num_hidden, int num_output,
		   int num_filtersInput, int num_filtersHidden,
		   double _minT, double _maxT) {

	algorithm = backprop;

	ni = num_input;
	nh = num_hidden;
	no = num_output;
	nfInput = num_filtersInput;
	nfHidden = num_filtersHidden;
	minT = _minT;
	maxT = _maxT;
	
	hiddenLayer = new Layer(nh,ni,nfInput,minT,maxT);
	outputLayer = new Layer(no,nh,nfHidden,minT,maxT);

	setLearningRate(0.001);

}

Deep_ICO::Deep_ICO(int num_input, int num_hidden, int num_output) {

	algorithm = backprop;

	ni = num_input;
	nh = num_hidden;
	no = num_output;
	nfInput = 0;
	nfHidden = 0;
	minT = 0;
	maxT = 0;
	
	hiddenLayer = new Layer(nh,ni);
	outputLayer = new Layer(no,nh);

	setLearningRate(0.001);

}

Deep_ICO::~Deep_ICO() {
	delete hiddenLayer;
	delete outputLayer;
}

void Deep_ICO::doStep(double* input, double* error) {

	switch (algorithm) {
	case backprop:
		hiddenLayer->setInputs(input);
		outputLayer->setErrors(error);
		// let's first calc all activities
		hiddenLayer->calcOutputs();
		// now that we have the outputs from the hidden layer
		// we can shovel them into the next layer
		for(int i=0;i<nh;i++) {
			// get the output of a neuron in the input layer
			double v = hiddenLayer->getNeuron(i)->getOutput();
			// set that output as an input to the next layer which
			// is distributed to all neurons
			outputLayer->setInput(i,v);
		}
		// now let's calc the output which can then be sent out
		outputLayer->calcOutputs();
		
		// Calcualte the errors for the hidden layer
		for(int i=0;i<hiddenLayer->getNneurons();i++) {
			double err = 0;
			for(int j=0;j<outputLayer->getNneurons();j++) {
				err = err + outputLayer->getNeuron(j)->getWeight(i) *
					outputLayer->getNeuron(j)->getError();
			}
			hiddenLayer->getNeuron(i)->setError(dsigm(hiddenLayer->getNeuron(i)->getOutput()) * err);
		}
		break;
	case ico:
		hiddenLayer->setInputs(input);
		hiddenLayer->setErrors(error);
		// let's first calc all activities
		hiddenLayer->calcOutputs();
		// now that we have the outputs from the hidden layer
		// we can shovel them into the next layer
		for(int i=0;i<nh;i++) {
			// get the output of a neuron in the input layer
			double v = hiddenLayer->getNeuron(i)->getOutput();
			// set that output as an input to the next layer which
			// is distributed to all neurons
			outputLayer->setInput(i,v);
		}
		// now let's calc the output which can then be sent out
		outputLayer->calcOutputs();
		
		// Calcualte the errors for the hidden layer
		for(int i=0;i<outputLayer->getNneurons();i++) {
			double err = 0;
			for(int j=0;j<hiddenLayer->getNneurons();j++) {
				err = err + hiddenLayer->getNeuron(j)->getWeight(i) *
					hiddenLayer->getNeuron(j)->getError();
			}
			outputLayer->getNeuron(i)->setError(dsigm(outputLayer->getNeuron(i)->getOutput()) * err);
		}
		break;		
	}

	outputLayer->doLearning();
	hiddenLayer->doLearning();
}

void Deep_ICO::setLearningRate(double rate) {
	hiddenLayer->setLearningRate(rate);
	outputLayer->setLearningRate(rate);
}


void Deep_ICO::initWeights(double max) {
	hiddenLayer->initWeights(max);
	outputLayer->initWeights(max);
}


void Deep_ICO::setUseDerivative(int useIt) {
	hiddenLayer->setUseDerivative(useIt);
	outputLayer->setUseDerivative(useIt);
}
