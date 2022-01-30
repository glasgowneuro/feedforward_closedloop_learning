#include "fcl.h"
#include <math.h>

/**
 * GNU GENERAL PUBLIC LICENSE
 * Version 3, 29 June 2007
 *
 * (C) 2017, Bernd Porr <bernd@glasgowneuro.tech>
 * (C) 2017, Paul Miller <paul@glasgowneuro.tech>
 **/

FeedforwardClosedloopLearning::FeedforwardClosedloopLearning(int num_input,
							     int* num_array,
							     int _num_layers
	) {
#ifdef DEBUG
	fprintf(stderr,"Creating instance of FeedforwardClosedloopLearning.\n");
#endif	
	ni = num_input;
	
	num_layers = _num_layers;
	n = new int[num_layers];
	layers = new Layer*[num_layers];

	// creating input layer
#ifdef DEBUG
	fprintf(stderr,"Creating input layer: ");
#endif
	layers[0] = new Layer(num_array[0], ni);
	n[0] = num_array[0];
#ifdef DEBUG
	fprintf(stderr,"n[0]=%d\n",n[0]);
#endif
	
	for(int i=1; i<num_layers; i++) {
		n[i] = num_array[i];
#ifdef DEBUG
		fprintf(stderr,"Creating layer %d: ",i);
#endif
		layers[i] = new Layer(n[i], layers[i-1]->getNneurons());
#ifdef DEBUG
		fprintf(stderr,"created with %d neurons.\n",layers[i]->getNneurons());
#endif
	}
	setLearningRate(0);
}

FeedforwardClosedloopLearning::~FeedforwardClosedloopLearning() {
	for (int i=0; i<num_layers; i++) {
		delete layers[i];
	}
	delete [] layers;
	delete [] n;
}


void FeedforwardClosedloopLearning::doStep(double* input, int n1, double* error, int n2) {
#ifdef DEBUG_FCL
	fprintf(stderr,"doStep: n1=%d,n2=%d\n",n1,n2);
#endif
	if (n1 != ni) {
		fprintf(stderr,"Input array dim mismatch: got: %d, want: %d\n",n1,ni);
		return;
	}
	if (n2 != layers[0]->getNneurons()) {
		fprintf(stderr,
			"Error array dim mismatch: got: %d, want: %d "
			"which is the number of neurons in the 1st hidden layer!\n",
			n2,layers[0]->getNneurons());
		return;
	}
	doStep(input,error);
}


void FeedforwardClosedloopLearning::setStep() {
	for (int k=0; k<num_layers; k++) {
		layers[k]->setStep(step);
	}
}

void FeedforwardClosedloopLearning::setActivationFunction(Neuron::ActivationFunction _activationFunction) {
	for (int k=0; k<num_layers; k++) {
		layers[k]->setActivationFunction(_activationFunction);
	}	
}

void FeedforwardClosedloopLearning::doLearning() {
	for (int k=0; k<num_layers; k++) {
		layers[k]->doLearning();
	}
}


void FeedforwardClosedloopLearning::setDecay(double decay) {
	for (int k=0; k<num_layers; k++) {
		layers[k]->setDecay(decay);
	}
}


void FeedforwardClosedloopLearning::doStep(double* input, double* error) {
	// we set the input to the input layer
	layers[0]->setInputs(input);
	// ..and calc its output
	layers[0]->calcOutputs();
	// new lets calc the other outputs
	for (int k=1; k<num_layers; k++) {
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
	// the error is injected into the 1st layer!
	for(int i=0;i<(layers[0]->getNneurons());i++) {
		layers[0]->getNeuron(i)->setError(error[i]);
	}
	for (int k=1; k<num_layers; k++) {
		Layer* emitterLayer = layers[k-1];
		Layer* receiverLayer = layers[k];
		// Calculate the errors for the hidden layer
		for(int i=0;i<receiverLayer->getNneurons();i++) {
			double norm = 0;
			double a = receiverLayer->getNeuron(i)->getManhattanNormOfWeightVector();
			a = a / receiverLayer->getNeuron(i)->getNinputs();
			norm += a;
			if (norm < 1E-30) norm = 1;
			double err = 0;
			for(int j=0;j<emitterLayer->getNneurons();j++) {
				err = err + receiverLayer->getNeuron(i)->getWeight(j) *
					emitterLayer->getNeuron(j)->getError();
#ifdef RANGE_CHECKS
				if (isnan(err) || (fabs(err)>10000) || (fabs(emitterLayer->getNeuron(j)->getError())>10000)) {
					printf("RANGE! FeedforwardClosedloopLearning::%s, step=%ld, j=%d, i=%d, hidLayerIndex=%d, "
					       "err=%e, emitterLayer->getNeuron(j)->getError()=%e\n",
					       __func__,step,j,i,k,err,emitterLayer->getNeuron(j)->getError());
				}
#endif
//				        if (fabs(err)>0) fprintf(stderr,"k=%d,i=%d,j=%d:err=%e\n",k,i,j,err);
			}
			err = err * learningRateDiscountFactor;
			err = err / norm;
			err = err * receiverLayer->getNeuron(i)->dActivation();
			receiverLayer->getNeuron(i)->setError(err);
		}
	}
	doLearning();
	setStep();
	step++;
}

void FeedforwardClosedloopLearning::setLearningRate(double rate) {
	for (int i=0; i<num_layers; i++) {
#ifdef DEBUG_FCL
		fprintf(stderr,"setLearningRate in layer %d\n",i);
#endif
		layers[i]->setLearningRate(rate);
	}
}

void FeedforwardClosedloopLearning::setMomentum(double momentum) {
	for (int i=0; i<num_layers; i++) {
#ifdef DEBUG_FCL
		fprintf(stderr,"setMomentum in layer %d\n",i);
#endif
		layers[i]->setMomentum(momentum);
	}
}



void FeedforwardClosedloopLearning::initWeights(double max, int initBias, Neuron::WeightInitMethod weightInitMethod) {
	for (int i=0; i<num_layers; i++) {
		layers[i]->initWeights(max,initBias,weightInitMethod);
	}
}


void FeedforwardClosedloopLearning::setUseDerivative(int useIt) {
	for (int i=0; i<num_layers; i++) {
		layers[i]->setUseDerivative(useIt);
	}
}

void FeedforwardClosedloopLearning::setBias(double _bias) {
	for (int i=0; i<num_layers; i++) {
		layers[i]->setBias(_bias);
	}
}

void FeedforwardClosedloopLearning::setDebugInfo() {
	for (int i=0; i<num_layers; i++) {
		layers[i]->setDebugInfo(i);
	}
}

// need to add bias weight
bool FeedforwardClosedloopLearning::saveModel(const char* name) {
	Layer *layer;
	Neuron *neuron;

	FILE *f=fopen(name, "wt");

	if(f) {
		for (int i=0; i<num_layers; i++) {
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
	Layer *layer;
	Neuron *neuron;
	double weight;
	int r;

	FILE *f=fopen(name, "r");

	if(f) {
		for (int i=0; i<num_layers; i++) {
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




/////////////////////



FeedforwardClosedloopLearningWithFilterbank::FeedforwardClosedloopLearningWithFilterbank(
			int num_of_inputs,
			int* num_of_neurons_per_layer_array,
			int num_layers,
			int num_filtersInput,
			double minT,
			double maxT) : FeedforwardClosedloopLearning(
				num_of_inputs * num_filtersInput,
				num_of_neurons_per_layer_array,
				num_layers) {
#ifdef DEBUG
	fprintf(stderr,"Creating instance of FeedforwardClosedloopLearningWithFilterbank.\n");
#endif	
	nFiltersPerInput = num_filtersInput;
	nInputs = num_of_inputs;
	bandpass = new Bandpass**[num_of_inputs];
	filterbankOutputs = new double[num_of_inputs * num_filtersInput];
	for(int i=0;i<num_of_inputs;i++) {
		bandpass[i] = new Bandpass*[num_filtersInput];
		double fs = 1;
		double fmin = fs/maxT;
		double fmax = fs/minT;
		double df = (fmax-fmin)/((double)(num_filtersInput-1));
		double f = fmin;
#ifdef DEBUG
		fprintf(stderr,"bandpass: fmin=%f,fmax=%f,df=%f\n",fmin,fmax,df);
#endif
		for(int j=0;j<num_filtersInput;j++) {
			bandpass[i][j] = new Bandpass();
#ifdef DEBUG
			fprintf(stderr,"bandpass[%d][%d]->setParameters(%f,%f)\n",
				i,j,fs,f);
#endif
			bandpass[i][j]->setParameters(f,dampingCoeff);
			f = f + df;
			for(int k=0;k<maxT;k++) {
				double a = 0;
				if (k==minT) {
					a = 1;
				}
				double b = bandpass[i][j]->filter(a);
				assert(b != NAN);
				assert(b != INFINITY);
			}
			bandpass[i][j]->reset();
		}
	}
}

FeedforwardClosedloopLearningWithFilterbank::~FeedforwardClosedloopLearningWithFilterbank() {
	delete[] filterbankOutputs;
	for(int i=0;i<nInputs;i++) {
		for(int j=0;j<nFiltersPerInput;j++) {
			delete bandpass[i][j];
		}
		delete[] bandpass[i];
	}
	delete[] bandpass;
}


void FeedforwardClosedloopLearningWithFilterbank::doStep(double* input, double* error) {
	for(int i=0;i<nInputs;i++) {
		for(int j=0;j<nFiltersPerInput;j++) {
			filterbankOutputs[i*nFiltersPerInput+j] = bandpass[i][j]->filter(input[i]);	
		}
	}
	FeedforwardClosedloopLearning::doStep(filterbankOutputs,error);
}
