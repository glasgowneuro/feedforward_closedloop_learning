#include "deep_feedback_learning.h"
#include <math.h>

DeepFeedbackLearning::DeepFeedbackLearning(int num_input, int* num_hidden_array, int num_output,
		 int num_filtersInput, int num_filtersHidden,
		 double _minT, double _maxT, int _num_hid_layers) {

	assert(_num_hid_layers>0);
	algorithm = backprop;

	ni = num_input;
	no = num_output;
	nfInput = num_filtersInput;
	nfHidden = num_filtersHidden;
	minT = _minT;
	maxT = _maxT;

	num_hid_layers = _num_hid_layers;
	n_hidden = new int[num_hid_layers];
	layers = new Layer*[num_hid_layers+1];

	int i=1;
	layers[0] = new Layer(num_hidden_array[0], ni,nfInput,minT,maxT);
	n_hidden[0] = num_hidden_array[0];

	for(i=1; i<num_hid_layers; i++) {
		n_hidden[i] = num_hidden_array[i];
		layers[i] = new Layer(n_hidden[i], n_hidden[i-1],nfHidden,minT,maxT);
	}

	layers[num_hid_layers] = new Layer(no, n_hidden[i-1],nfHidden,minT,maxT);
	outputLayer = layers[num_hid_layers];

	setLearningRate(0.001);

}

DeepFeedbackLearning::DeepFeedbackLearning(int num_input, int* num_hidden_array, int num_output, int _num_hid_layers) {

	algorithm = backprop;

	ni = num_input;
	no = num_output;
	nfInput = 0;
	nfHidden = 0;
	minT = 0;
	maxT = 0;
	
	num_hid_layers = _num_hid_layers;
	n_hidden = new int[num_hid_layers];
	layers = new Layer*[num_hid_layers+1];

	int i=1;
	layers[0] = new Layer(num_hidden_array[0], ni);
	n_hidden[0] = num_hidden_array[0];
	for(i=1; i<num_hid_layers-1; i++) {
		n_hidden[i] = num_hidden_array[i];
		layers[i] = new Layer(n_hidden[i], n_hidden[i-1]);
	}
	layers[num_hid_layers] = new Layer(no, n_hidden[i-1]);
	outputLayer = layers[num_hid_layers];

	setLearningRate(0.001);

}

DeepFeedbackLearning::~DeepFeedbackLearning() {
	for (int i=0; i<num_hid_layers+1; i++) {
		delete layers[i];
	}
}

void DeepFeedbackLearning::doStep(double* input, double* error) {
	Layer *hiddenLayer, *nextLayer;
	int nh;
	int nFiltersInput = 10;
	int nFiltersHidden = 10;

	switch (algorithm) {
	case backprop:
		for (int i=0; i<num_hid_layers; i++) {
			nh = n_hidden[i];
			hiddenLayer = layers[i];
			nextLayer = layers[i+1];
			hiddenLayer->setInputs(input);
			nextLayer->setErrors(error);
			// let's first calc all activities
			hiddenLayer->calcOutputs();
			// now that we have the outputs from the hidden layer
			// we can shovel them into the next layer
			for(int i=0;i<nh;i++) {
				// get the output of a neuron in the input layer
				double v = hiddenLayer->getNeuron(i)->getOutput();
				// set that output as an input to the next layer which
				// is distributed to all neurons
				nextLayer->setInput(i,v);
			}

			// now let's calc the output which can then be sent out
			nextLayer->calcOutputs();

			// Calculate the errors for the hidden layer
			for(int i=0;i<hiddenLayer->getNneurons();i++) {
				double err = 0;
				for(int j=0;j<nextLayer->getNneurons();j++) {
					err = err + nextLayer->getNeuron(j)->getWeight(i) *
							nextLayer->getNeuron(j)->getError();
					if (isnan(err)) {
						fprintf(stderr,"doStep: layer: %d err=%f, nextLayer->getNeuron(j)->getWeight(i)=%f, nextoutputLayer->getNeuron(j)->getError()=%f\n",
								i,err,outputLayer->getNeuron(j)->getWeight(i), outputLayer->getNeuron(j)->getError());
						exit(0);
					}
				}
				hiddenLayer->getNeuron(i)->setError(dsigm(hiddenLayer->getNeuron(i)->getOutput()) * err);
			}
		}
		break;
	case ico:
		for (int i=0; i<num_hid_layers; i++) {
			nh = n_hidden[i];
			hiddenLayer = layers[i];
			nextLayer = layers[i+1];

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
				nextLayer->setInput(i,v);
			}
			// now let's calc the output which can then be sent out
			nextLayer->calcOutputs();

			// Calculate the errors for the hidden layer
			for(int i=0;i<nextLayer->getNneurons();i++) {
				double err = 0;
				for(int j=0;j<hiddenLayer->getNneurons();j++) {
					err = err + hiddenLayer->getNeuron(j)->getWeight(i) *
							hiddenLayer->getNeuron(j)->getError();
				}
				nextLayer->getNeuron(i)->setError(dsigm(nextLayer->getNeuron(i)->getOutput()) * err);
			}
		}
		break;		
	}

	for (int i=num_hid_layers; i>-1; i--) {
		layers[i]->doLearning();
	}
}

void DeepFeedbackLearning::setLearningRate(double rate) {
	for (int i=0; i<num_hid_layers+1; i++) {
		layers[i]->setLearningRate(rate);
	}
}


void DeepFeedbackLearning::initWeights(double max) {
	for (int i=0; i<num_hid_layers+1; i++) {
		layers[i]->initWeights(max);
	}
}


void DeepFeedbackLearning::setUseDerivative(int useIt) {
	for (int i=0; i<num_hid_layers+1; i++) {
		layers[i]->setUseDerivative(useIt);
	}
}
