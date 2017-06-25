#include "deep_feedback_learning.h"
#include <math.h>

/**
 * GNU GENERAL PUBLIC LICENSE
 * Version 3, 29 June 2007
 *
 * (C) 2017, Bernd Porr <mail@berndporr.me.uk>, <bernd@glasgowneuro.tech>
 * (C) 2017, Paul Miller <nlholdem@hotmail.com>
 **/

DeepFeedbackLearning::DeepFeedbackLearning(int num_input, int* num_hidden_array, int _num_hid_layers, int num_output,
					   int num_filtersInput, int num_filtersHidden,
					   double _minT, double _maxT) {

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

	// creating input layer
#ifdef DEBUG_DFL
	fprintf(stderr,"Creating input layer: ");
#endif
	layers[0] = new Layer(num_hidden_array[0], ni,nfInput,minT,maxT);
	n_hidden[0] = num_hidden_array[0];
#ifdef DEBUG_DFL
	fprintf(stderr,"created! n_hidden[0]=%d\n",n_hidden[0]);
#endif

#ifdef DEBUG_DFL
	fprintf(stderr,"Creating hidden layers: ");
#endif
	// additional hidden layers
	// note that these are _additional_ layers
	for(int i=1; i<num_hid_layers; i++) {
		n_hidden[i] = num_hidden_array[i];
#ifdef DEBUG_DFL
		fprintf(stderr,"Creating layers %d: ",i);
#endif
		layers[i] = new Layer(n_hidden[i], layers[i-1]->getNneurons(),nfHidden,minT,maxT);
#ifdef DEBUG_DFL
		fprintf(stderr,"created with %d neurons.",layers[i]->getNneurons());
#endif

	}

	// output layer
#ifdef DEBUG_DFL
	fprintf(stderr,"Creating output layer: ");
#endif
	layers[num_hid_layers] = new Layer(no, layers[num_hid_layers-1]->getNneurons(),nfHidden,minT,maxT);
#ifdef DEBUG_DFL
	fprintf(stderr,"created! n_hidden[0]=%d\n",layers[i]->getNneurons());
#endif

	setLearningRate(0);
}

DeepFeedbackLearning::DeepFeedbackLearning(int num_input, int* num_hidden_array, int _num_hid_layers, int num_output) {

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

	// input layer 
	layers[0] = new Layer(num_hidden_array[0], ni);
	n_hidden[0] = num_hidden_array[0];
	
	for(int i=1; i<num_hid_layers-1; i++) {
		n_hidden[i] = num_hidden_array[i];
		layers[i] = new Layer(n_hidden[i], layers[i-1]->getNneurons());
	}
	layers[num_hid_layers] = new Layer(no, layers[num_hid_layers-1]->getNneurons());

	setLearningRate(0);
}

DeepFeedbackLearning::~DeepFeedbackLearning() {
	for (int i=0; i<num_hid_layers+1; i++) {
		delete layers[i];
	}
	delete [] layers;
	delete [] n_hidden;
}


void DeepFeedbackLearning::doStep(double* input, int n1, double* error, int n2) {
#ifdef DEBUG_DFL
		fprintf(stderr,"doStep: n1=%d,n2=%d\n",n1,n2);
#endif
		if (n1 != ni) {
			fprintf(stderr,"Input array dim mismatch: got: %d, want: %d\n",n1,ni);
			return;
		}
		switch (algorithm) {
		case backprop:
			if (n2 != no) {
				fprintf(stderr,"Error array dim mismatch: got: %d, want: %d\n",n2,no);
				return;
			}
			break;
		case ico:
			if (n2 != ni) {
				fprintf(stderr,"Error array dim mismatch: got: %d, want: %d\n",n2,ni);
				return;
			}
			break;
		}
		doStep(input,error);
	}


void DeepFeedbackLearning::doStep(double* input, double* error) {
	switch (algorithm) {
	case backprop:
		// we set the input to the input layer
		layers[0]->setInputs(input);
		// ..and calc its output
		layers[0]->calcOutputs();
		// new lets calc the other outputs
		for (int i=1; i<=num_hid_layers; i++) {
			Layer* emitterLayer = layers[i-1];
			Layer* receiverLayer = layers[i];
			// now that we have the outputs from the previous layer
			// we can shovel them into the next layer
			for(int i=0;i<emitterLayer->getNneurons();i++) {
				// get the output of a neuron in the input layer
				double v = emitterLayer->getNeuron(i)->getOutput();
				// set that output as an input to the next layer which
				// is distributed to all neurons
				receiverLayer->setInput(i,v);
			}
			
			// now let's calc the output which can then be sent out
			receiverLayer->calcOutputs();
	        }
		// error processing
		layers[num_hid_layers]->setErrors(error);
		for (int i=num_hid_layers; i>0; i--) {
			Layer* emitterLayer = layers[i];
			Layer* receiverLayer = layers[i-1];
			// Calculate the errors for the hidden layer
			for(int i=0;i<receiverLayer->getNneurons();i++) {
				double err = 0;
				for(int j=0;j<emitterLayer->getNneurons();j++) {
					err = err + emitterLayer->getNeuron(j)->getWeight(i) *
						emitterLayer->getNeuron(j)->getError();
					if (isnan(err)) {
		                                printf("err=nan\n");
						exit(0);
					}
				}
				receiverLayer->getNeuron(i)->setError(dsigm(receiverLayer->getNeuron(i)->getOutput()) * err);
			}
	        }
		break;
	case ico:
		// we set the input to the input layer
		layers[0]->setInputs(input);
		// ..and calc its output
		layers[0]->calcOutputs();
		// new lets calc the other outputs
		for (int i=1; i<=num_hid_layers; i++) {
			Layer* emitterLayer = layers[i-1];
			Layer* receiverLayer = layers[i];
			// now that we have the outputs from the previous layer
			// we can shovel them into the next layer
			for(int i=0;i<emitterLayer->getNneurons();i++) {
				// get the output of a neuron in the input layer
				double v = emitterLayer->getNeuron(i)->getOutput();
				// set that output as an input to the next layer which
				// is distributed to all neurons
				receiverLayer->setInput(i,v);
			}
			
			// now let's calc the output which can then be sent out
			receiverLayer->calcOutputs();
	        }
		// error processing
		layers[0]->setErrors(error);
		for (int i=1; i<=num_hid_layers; i++) {
			Layer* emitterLayer = layers[i-1];
			Layer* receiverLayer = layers[i];
			// Calculate the errors for the hidden layer
			for(int i=0;i<receiverLayer->getNneurons();i++) {
				double err = 0;
				for(int j=0;j<emitterLayer->getNneurons();j++) {
					err = err + emitterLayer->getNeuron(j)->getWeight(i) *
						emitterLayer->getNeuron(j)->getError();
					if (isnan(err)) {
		                                printf("err=nan\n");
						exit(0);
					}
				}
				receiverLayer->getNeuron(i)->setError(dsigm(receiverLayer->getNeuron(i)->getOutput()) * err);
				receiverLayer->getNeuron(i)->setError(err);
			}
	        }
		break;
	}

	for (int i=num_hid_layers; i>-1; i--) {
		layers[i]->doLearning();
	}
}

void DeepFeedbackLearning::setLearningRate(double rate) {
	for (int i=0; i<(num_hid_layers+1); i++) {
#ifdef DEBUG_DFL
		fprintf(stderr,"setLearningRate in layer %d\n",i);
#endif
		layers[i]->setLearningRate(rate);
	}
}


void DeepFeedbackLearning::initWeights(double max) {
	for (int i=0; i<(num_hid_layers+1); i++) {
		layers[i]->initWeights(max);
	}
}


void DeepFeedbackLearning::setUseDerivative(int useIt) {
	for (int i=0; i<(num_hid_layers+1); i++) {
		layers[i]->setUseDerivative(useIt);
	}
}
