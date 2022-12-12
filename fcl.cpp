#include "fcl.h"
#include <math.h>

/**
 * GNU GENERAL PUBLIC LICENSE
 * Version 3, 29 June 2007
 *
 * (C) 2017-2022, Bernd Porr <bernd@glasgowneuro.tech>
 * (C) 2017, Paul Miller <paul@glasgowneuro.tech>
 **/

FeedforwardClosedloopLearning::FeedforwardClosedloopLearning(const int num_input,
							     const std::vector<int> &num_of_neurons_per_layer) {
#ifdef DEBUG
	fprintf(stderr,"Creating instance of FeedforwardClosedloopLearning.\n");
#endif	

	n_neurons_per_layer = num_of_neurons_per_layer;
	layers = new FCLLayer*[n_neurons_per_layer.size()];
	ni = (unsigned)num_input;

	// creating input layer
#ifdef DEBUG
	fprintf(stderr,"Creating input layer: ");
#endif
	layers[0] = new FCLLayer(n_neurons_per_layer[0], ni);
#ifdef DEBUG
	fprintf(stderr,"n[0]=%d\n",n_neurons_per_layer[0]);
#endif
	
	for(unsigned i=1; i<n_neurons_per_layer.size(); i++) {
#ifdef DEBUG
		fprintf(stderr,"Creating layer %d: ",i);
#endif
		layers[i] = new FCLLayer(n_neurons_per_layer[i], layers[i-1]->getNneurons());
#ifdef DEBUG
		fprintf(stderr,"created with %d neurons.\n",layers[i]->getNneurons());
#endif
	}
	setLearningRate(0);
}

FeedforwardClosedloopLearning::~FeedforwardClosedloopLearning() {
	for (unsigned i=0; i<n_neurons_per_layer.size(); i++) {
		delete layers[i];
	}
	delete [] layers;
}


void FeedforwardClosedloopLearning::setStep() {
	for (unsigned k=0; k<n_neurons_per_layer.size(); k++) {
		layers[k]->setStep(step);
	}
}

void FeedforwardClosedloopLearning::setActivationFunction(FCLNeuron::ActivationFunction _activationFunction) {
	for (unsigned k=0; k<n_neurons_per_layer.size(); k++) {
		layers[k]->setActivationFunction(_activationFunction);
	}	
}

void FeedforwardClosedloopLearning::doLearning() {
	for (unsigned k=0; k<n_neurons_per_layer.size(); k++) {
		layers[k]->doLearning();
	}
}


void FeedforwardClosedloopLearning::setDecay(double decay) {
	for (unsigned k=0; k<n_neurons_per_layer.size(); k++) {
		layers[k]->setDecay(decay);
	}
}


void FeedforwardClosedloopLearning::doStep(const std::vector<double> &input, const std::vector<double> &error) {
	if (input.size() != ni) {
		char tmp[256];
		sprintf(tmp,"Input array dim mismatch: got: %ld, want: %d.",input.size(),ni);
		#ifdef DEBUG
		fprintf(stderr,"%s\n",tmp);
		#endif
		throw tmp;
	}
	if (error.size() != ni) {
		char tmp[256];
		sprintf(tmp,
			"Error array dim mismatch: got: %ld, want: %d "
			"which is the number of neurons in the 1st hidden layer!",
			error.size(),layers[0]->getNneurons());
		#ifdef DEBUG
		fprintf(stderr,"%s\n",tmp);
		#endif
		throw tmp;
	}
	// we set the input to the input layer
	layers[0]->setInputs(input.data());
	// ..and calc its output
	layers[0]->calcOutputs();
	// new lets calc the other outputs
	for (unsigned k=1; k<n_neurons_per_layer.size(); k++) {
		FCLLayer* emitterLayer = layers[k-1];
		FCLLayer* receiverLayer = layers[k];
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
	// the error is injected into the 1st layer!
	for(int i=0;i<(layers[0]->getNneurons());i++) {
		layers[0]->getNeuron(i)->setError(error[i]);
	}
	for (unsigned k=1; k<n_neurons_per_layer.size(); k++) {
		FCLLayer* emitterLayer = layers[k-1];
		FCLLayer* receiverLayer = layers[k];
		// Calculate the errors for the hidden layer
		for(int i=0;i<receiverLayer->getNneurons();i++) {
			double err = 0;
			for(int j=0;j<emitterLayer->getNneurons();j++) {
				err = err + receiverLayer->getNeuron(i)->getWeight(j) *
					emitterLayer->getNeuron(j)->getError();
#ifdef DEBUG
				if (isnan(err) || (fabs(err)>10000) || (fabs(emitterLayer->getNeuron(j)->getError())>10000)) {
					printf("RANGE! FeedforwardClosedloopLearning::%s, step=%ld, j=%d, i=%d, hidLayerIndex=%d, "
					       "err=%e, emitterLayer->getNeuron(j)->getError()=%e\n",
					       __func__,step,j,i,k,err,emitterLayer->getNeuron(j)->getError());
				}
#endif
			}
			err = err * learningRateDiscountFactor;
			err = err * emitterLayer->getNneurons();
			err = err * receiverLayer->getNeuron(i)->dActivation();
			receiverLayer->getNeuron(i)->setError(err);
		}
	}
	doLearning();
	setStep();
	step++;
}

void FeedforwardClosedloopLearning::setLearningRate(double rate) {
	for (unsigned i=0; i<n_neurons_per_layer.size(); i++) {
#ifdef DEBUG_FCL
		fprintf(stderr,"setLearningRate in layer %d\n",i);
#endif
		layers[i]->setLearningRate(rate);
	}
}

void FeedforwardClosedloopLearning::setMomentum(double momentum) {
	for (unsigned i=0; i<n_neurons_per_layer.size(); i++) {
#ifdef DEBUG_FCL
		fprintf(stderr,"setMomentum in layer %d\n",i);
#endif
		layers[i]->setMomentum(momentum);
	}
}



void FeedforwardClosedloopLearning::initWeights(double max, int initBias, FCLNeuron::WeightInitMethod weightInitMethod) {
	for (unsigned i=0; i<n_neurons_per_layer.size(); i++) {
		layers[i]->initWeights(max,initBias,weightInitMethod);
	}
}


void FeedforwardClosedloopLearning::setBias(double _bias) {
	for (unsigned i=0; i<n_neurons_per_layer.size(); i++) {
		layers[i]->setBias(_bias);
	}
}

void FeedforwardClosedloopLearning::setDebugInfo() {
	for (unsigned i=0; i<n_neurons_per_layer.size(); i++) {
		layers[i]->setDebugInfo(i);
	}
}

// need to add bias weight
bool FeedforwardClosedloopLearning::saveModel(const char* name) {
	FCLLayer *layer;
	FCLNeuron *neuron;

	FILE *f=fopen(name, "wt");

	if(f) {
		for (unsigned i=0; i<n_neurons_per_layer.size(); i++) {
			layer = layers[i];
			for (int j=0; j<layer->getNneurons(); j++) {
				neuron = layer->getNeuron(j);
				for (int k=0; k<neuron->getNinputs(); k++) {
					if(neuron->getMask(k)) {
						fprintf(f, "%.16lf ", neuron->getWeight(k));
					}
				}
				fprintf(f, "%.16lf ", neuron->getBiasWeight());
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

bool FeedforwardClosedloopLearning::loadModel(const char* name) {
	FCLLayer *layer;
	FCLNeuron *neuron;
	double weight;
	int r;

	FILE *f=fopen(name, "r");

	if(f) {
		for (unsigned i=0; i<n_neurons_per_layer.size(); i++) {
			layer = layers[i];
			for (int j=0; j<layer->getNneurons(); j++) {
				neuron = layer->getNeuron(j);
				for (int k=0; k<neuron->getNinputs(); k++) {
					if(neuron->getMask(k)) {
						r = fscanf(f, "%lf ", &weight);
						if (r < 0) return false;
						neuron->setWeight(k, weight);						
					}
				}
				r = fscanf(f, "%lf", &weight);
				if (r < 0) return false;
				neuron->setBiasWeight(weight);
				r = fscanf(f, "%*c");
				if (r < 0) return false;
			}
			r = fscanf(f, "%*c");
			if (r < 0) return false;
		}
		r = fscanf(f, "%*c");
		if (r < 0) return false;
	}
	else {
		return false;
	}

	fclose(f);
	return true;
}
