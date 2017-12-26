#include "deep_feedback_learning.h"
#include <math.h>

/**
 * GNU GENERAL PUBLIC LICENSE
 * Version 3, 29 June 2007
 *
 * (C) 2017, Bernd Porr <bernd@glasgowneuro.tech>
 * (C) 2017, Paul Miller <paul@glasgowneuro.tech>
 **/

DeepFeedbackLearning::DeepFeedbackLearning(
	int num_of_inputs,
	int* num_of_hidden_neurons_per_layer_array,
	int _num_hid_layers,
	int num_outputs,
	int num_filtersInput,
	int num_filtersHidden,
	double _minT,
	double _maxT) {
	
	assert(_num_hid_layers>0);
	algorithm = backprop;

	ni = num_of_inputs;
	no = num_outputs;
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
	layers[0] = new Layer(num_of_hidden_neurons_per_layer_array[0], ni,nfInput,minT,maxT);
	n_hidden[0] = num_of_hidden_neurons_per_layer_array[0];
#ifdef DEBUG_DFL
	fprintf(stderr,"created! n_hidden[0]=%d\n",n_hidden[0]);
#endif

#ifdef DEBUG_DFL
	fprintf(stderr,"Creating hidden layers: ");
#endif
	// additional hidden layers
	// note that these are _additional_ layers
	for(int i=1; i<num_hid_layers; i++) {
		n_hidden[i] = num_of_hidden_neurons_per_layer_array[i];
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
	setDebugInfo();
}

DeepFeedbackLearning::DeepFeedbackLearning(int num_input,
					   int* num_hidden_array,
					   int _num_hid_layers,
					   int num_output) {

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

	// creating input layer
#ifdef DEBUG_DFL
	fprintf(stderr,"Creating input layer: ");
#endif
	layers[0] = new Layer(num_hidden_array[0], ni);
	n_hidden[0] = num_hidden_array[0];
#ifdef DEBUG_DFL
	fprintf(stderr,"created! n_hidden[0]=%d\n",n_hidden[0]);
#endif
	
	for(int i=1; i<num_hid_layers; i++) {
		n_hidden[i] = num_hidden_array[i];
#ifdef DEBUG_DFL
		fprintf(stderr,"Creating layers %d: ",i);
#endif
		layers[i] = new Layer(n_hidden[i], layers[i-1]->getNneurons());
#ifdef DEBUG_DFL
		fprintf(stderr,"created with %d neurons.",layers[i]->getNneurons());
#endif
	}
	// output layer
	#ifdef DEBUG_DFL
		fprintf(stderr,"Creating output layer: ");
	#endif
	layers[num_hid_layers] = new Layer(no, layers[num_hid_layers-1]->getNneurons());
#ifdef DEBUG_DFL
	fprintf(stderr,"created! n_hidden[0]=%d\n",layers[i]->getNneurons());
#endif

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
			doStepBackprop(input,error);
			break;
		case ico:
			if (n2 != layers[0]->getNneurons()) {
				fprintf(stderr,
					"Error array dim mismatch: got: %d, want: %d "
					"which is the number of neurons in the 1st hidden layer!\n",
					n2,layers[0]->getNneurons());
				return;
			}
			doStepForwardprop(input,error);
			break;
		}
	}


void DeepFeedbackLearning::doStep(double* input, double* error) {
#ifdef DEBUG_DFL
		fprintf(stderr,"doStep: n1=%d,n2=%d\n",n1,n2);
#endif
		switch (algorithm) {
		case backprop:
			doStepBackprop(input,error);
			break;
		case ico:
			doStepForwardprop(input,error);
			break;
		}
	}


void DeepFeedbackLearning::setStep() {
	for (int k=0; k<=num_hid_layers; k++) {
		layers[k]->setStep(step);
	}
}

void DeepFeedbackLearning::doLearning() {
	for (int k=0; k<=num_hid_layers; k++) {
		layers[k]->doLearning();
	}
}


void DeepFeedbackLearning::doStepBackprop(double* input, double* error) {
	// Let's first propagate the signal through the layers
	// we set the input to the input layer
	layers[0]->setInputs(input);
	// ..and calc its output
	layers[0]->calcOutputs();
	// new lets calc the other outputs
	for (int k=1; k<=num_hid_layers; k++) {
		// This layer generates the output
		Layer* emitterLayer = layers[k-1];
		Layer* receiverLayer = layers[k];
		// now that we have the outputs from the previous layer
		// we can shovel them into the next layer
		// loop through all neurons and copy the content to the next layer
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
	// we put the error in the last layer, the output layer
	for (int i=0; i<layers[num_hid_layers]->getNneurons(); i++) {
		layers[num_hid_layers]->setError(i,error[i] * layers[num_hid_layers]->getNeuron(i)->dActivation());
	}
	// let's now loop through the layers backwards
	for (int k=num_hid_layers; k>0; k--) {
		// the layer which has the error which is further down towards the
		// output
		Layer* emitterLayer = layers[k];
		// the layer which receives the error is the one which is towards
		// the input
		Layer* receiverLayer = layers[k-1];
		// Calculate the errors for the hidden layers and the input layer
		// loop through all neurons of the receiver layer and set their
		// errors
		for(int i=0;i<receiverLayer->getNneurons();i++) {
			// accumulate the error by looping through the
			// emitter layer, get their errors and weight them
			// with the corresponding weights leading to that
			// neuron
			double err = 0;
			for(int j=0;j<emitterLayer->getNneurons();j++) {
				// that is the error from neuron j in the emitter
				// layer influencing the error in the receiver
				// layer i weighted by its corresponding weight
				err = err + emitterLayer->getNeuron(j)->getAvgWeight(i);
				// sanity check that it's not NAN
				assert(!isnan(err));
				//fprintf(stderr,"k=%d,i=%d,j=%d:err=%e\n",k,i,j,err);
			}
			receiverLayer->getNeuron(i)->setError(receiverLayer->getNeuron(i)->dActivation() * err);
		}
	}
	doLearning();
	setStep();
	step++;
}

void DeepFeedbackLearning::doStepForwardprop(double* input, double* error) {
	// we set the input to the input layer
	layers[0]->setInputs(input);
	// ..and calc its output
	layers[0]->calcOutputs();
	// new lets calc the other outputs
	for (int k=1; k<=num_hid_layers; k++) {
		Layer* emitterLayer = layers[k-1];
		Layer* receiverLayer = layers[k];
		// now that we have the outputs from the previous layer
		// we can shovel them into the next layer
		for(int j=0;j<emitterLayer->getNneurons();j++) {
			// get the output of a neuron in the input layer
			double v = emitterLayer->getNeuron(j)->getOutput();
			// set that output as an input to the next layer which
			// is distributed to all neurons
			receiverLayer->setInput(j,v);
		}
		
		// now let's calc the output which can then be sent out
		receiverLayer->calcOutputs();
	}
	// the error is injected into the 1st _hidden_ layer!
	layers[0]->setErrors(error);
	for (int k=0; k<num_hid_layers; k++) {
		Layer* emitterLayer = layers[k];
		Layer* receiverLayer = layers[k+1];
		// Calculate the errors for the hidden layer
		for(int i=0;i<receiverLayer->getNneurons();i++) {
			double err = 0;
			for(int j=0;j<emitterLayer->getNneurons();j++) {
				err = err + receiverLayer->getNeuron(i)->getAvgWeight(j) *
					emitterLayer->getNeuron(j)->getError();
#ifdef RANGE_CHECKS
				if (isnan(err) || (fabs(err)>100) || (fabs(emitterLayer->getNeuron(j)->getError())>100)) {
					printf("DeepFeedbackLearning::%s, step=%ld, j=%d, i=%d, hidLayerIndex=%d, "
					       "err=%e, emitterLayer->getNeuron(j)->getError()=%e\n",
					       __func__,step,j,i,k,err,emitterLayer->getNeuron(j)->getError());
				}
#endif
//				if (fabs(err)>0) fprintf(stderr,"k=%d,i=%d,j=%d:err=%e\n",k,i,j,err);
			}
			err = err * learningRateDiscountFactor;
			receiverLayer->getNeuron(i)->setError(err * receiverLayer->getNeuron(i)->dActivation());
		}
	}
	doLearning();
	setStep();
	step++;
}

void DeepFeedbackLearning::setLearningRate(double rate) {
	for (int i=0; i<(num_hid_layers+1); i++) {
#ifdef DEBUG_DFL
		fprintf(stderr,"setLearningRate in layer %d\n",i);
#endif
		layers[i]->setLearningRate(rate);
	}
}

void DeepFeedbackLearning::setMomentum(double momentum) {
	for (int i=0; i<(num_hid_layers+1); i++) {
#ifdef DEBUG_DFL
		fprintf(stderr,"setMomentum in layer %d\n",i);
#endif
		layers[i]->setMomentum(momentum);
	}
}



void DeepFeedbackLearning::initWeights(double max, int initBias, Neuron::WeightInitMethod weightInitMethod) {
	for (int i=0; i<(num_hid_layers+1); i++) {
		layers[i]->initWeights(max,initBias,weightInitMethod);
	}
}


void DeepFeedbackLearning::setUseDerivative(int useIt) {
	for (int i=0; i<(num_hid_layers+1); i++) {
		layers[i]->setUseDerivative(useIt);
	}
}

void DeepFeedbackLearning::setBias(double _bias) {
	for (int i=0; i<(num_hid_layers+1); i++) {
		layers[i]->setBias(_bias);
	}
}

void DeepFeedbackLearning::setDebugInfo() {
	for (int i=0; i<(num_hid_layers+1); i++) {
		layers[i]->setDebugInfo(i);
	}
}

// need to add bias weight
bool DeepFeedbackLearning::saveModel(const char* name) {
	Layer *layer;
	Neuron *neuron;
	double weight;

	FILE *f=fopen(name, "wt");

	if(f) {
		for (int i=0; i<num_hid_layers+1; i++) {
			layer = layers[i];
			for (int j=0; j<layer->getNneurons(); j++) {
				neuron = layer->getNeuron(j);
				for (int k=0; k<neuron->getNinputs(); k++) {
					if(neuron->getMask(k)) {
						for (int l=0; l<neuron->getNfilters(); l++) {
							fprintf(f, "%lf ", neuron->getWeight(k,l));
						}
					}
				}
				fprintf(f, "%lf ", neuron->getBiasWeight());
				fprintf(f, "\n");
			}
			fprintf(f, "\n");
		}
		fprintf(f, "\n");
	}
	else {
		return false;
	}

	fclose(f);
	return true;
}

bool DeepFeedbackLearning::loadModel(const char* name) {
	Layer *layer;
	Neuron *neuron;
	double weight;
	int result;

	FILE *f=fopen(name, "r");

	if(f) {
		for (int i=0; i<num_hid_layers+1; i++) {
			layer = layers[i];
			for (int j=0; j<layer->getNneurons(); j++) {
				neuron = layer->getNeuron(j);
				for (int k=0; k<neuron->getNinputs(); k++) {
					if(neuron->getMask(k)) {
						for (int l=0; l<neuron->getNfilters(); l++) {
							result=fscanf(f, "%lf ", &weight);
							neuron->setWeight(k, weight, l);
						}
					}
				}
				result=fscanf(f, "%lf", &weight);
				neuron->setBiasWeight(weight);
				result=fscanf(f, "%*c");
			}
			result=fscanf(f, "%*c");
		}
		result=fscanf(f, "%*c");
	}
	else {
		return false;
	}

	fclose(f);
	return true;
}
