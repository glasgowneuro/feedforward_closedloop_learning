#include "neuron.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

/**
 * GNU GENERAL PUBLIC LICENSE
 * Version 3, 29 June 2007
 *
 * (C) 2017-2022, Bernd Porr <bernd@glasgowneuro.tech>
 * (C) 2017, Paul Miller <paul@glasgowneuro.tech>
 **/

#define SUM_ERR_THRES 1000

Neuron::Neuron(int _nInputs) {
	nInputs = _nInputs;

	mask = new unsigned char[nInputs];
	weights = new double[nInputs];
	initialWeights = new double[nInputs];
	weightChange = new double[nInputs];
	inputs = new double[nInputs];
	sum = 0;
	output = 0;
	error = 0;
	learningRate = 0;
	for(int i=0;i<nInputs;i++) {
		weights[i] = 0;
		initialWeights[i] = 0;
		weightChange[i] = 0;
		inputs[i] = 0;
		mask[i] = 1;
	}	
}

Neuron::~Neuron() {
	delete [] weights;
	delete [] initialWeights;
	delete [] weightChange;
	delete [] inputs;
	delete [] mask;
}


void Neuron::calcOutput() {
	double* weightsp = weights;
	double* inputp = inputs;
	unsigned char * maskp = mask;

	// global variable
	sum = 0;

	for(int i=0;i<nInputs;i++) {
		// checking indexing
		assert((mask+i) == maskp);
		assert((weights+i) == weightsp);
		assert((inputs+i) == inputp);
		if (*maskp) {
			// checking values
			assert(weights[i] == (*weightsp));
			assert(inputs[i] == (*inputp));
			sum = sum + (*weightsp) * (*inputp);
#ifdef DEBUG
			if (isnan(sum) || isnan(weights[i]) || isnan(inputs[i]) || (fabs(sum)>SUM_ERR_THRES)) {
				fprintf(stderr,"Out of range Neuron::%s step=%ld, L=%d, N=%d, %f, %f, %f, %d\n",
					__func__,step,layerIndex,neuronIndex,sum,weights[i],inputs[i],i);
			}
#endif
		}
		weightsp++;
		inputp++;
		maskp++;
	}
	sum = sum + biasweight * bias;

#ifdef DEBUG
	if (fabs(sum) > SUM_ERR_THRES) fprintf(stderr,"Neuron::%s, Sum (%e) is very high in layer %d, neuron %d, step %ld.\n",__func__,sum,layerIndex,neuronIndex,step);
#endif
	
	switch (activationFunction) {
	case LINEAR:
		output = sum;
		break;
	case TANH:
	case TANHLIMIT:
		output = tanh(sum);
		break;
	case RELU:
		if (sum>0) {
			output = sum;
		} else {
			output = 0;
		}
		break;
	case REMAXLU:
		if (sum>0) {
			if (sum<1) {
				output = sum;
			} else {
				output = 1;
			}
		} else {
			output = 0;
		}
		break;
	default:
		output = sum;	
	}
}


double Neuron::dActivation() {
	double d;
	switch (activationFunction) {
	case LINEAR:
		return 1;
	case TANH:
		d = (1.0 - output*output);
		return d;
		break;
	case TANHLIMIT:
		d = (1.0 - fabs(output*output*output));
		if (d<0) return 0;
		return d;
		break;
	case RELU:
		if (output>0) {
			return 1;
		} else {
			return 0;
		}
		break;
	case REMAXLU:
		if ((output>0)&&(output<1)) {
			return 1;
		} else {
			return 0;
		}
		break;		
		
	default:
		return 1;
	}
}



void Neuron::doLearning() {
	double* inputsp = inputs;
	double* weightsp = weights;
	double* weightschp = weightChange;
	unsigned char * maskp = mask;
	maxDet = 0;
	for(int i=0;i<nInputs;i++) {
		assert((mask+i) == maskp);
		assert((weights+i) == weightsp);
		assert((inputs+i) == inputsp);
		assert((weightChange+i) == weightschp);
		if (*maskp) {
			*weightschp = momentum * (*weightschp) +
				(*inputsp) * error * learningRate * learningRateFactor -
				(*weightsp) * decay * learningRate * fabs(error);
			*weightsp = *weightsp + *weightschp;
#ifdef DEBUG
			if (isnan(sum) || isnan(weights[i]) || isnan(inputs[i]) || (fabs(sum)>SUM_ERR_THRES)) {
				fprintf(stderr,"Out of range Neuron::%s step=%ld, L=%d, N=%d, %f, %f, %f, %d\n",
					__func__,step,layerIndex,neuronIndex,sum,weights[i],inputs[i],i);
			}
#endif
		}
		inputsp++;
		maskp++;
		weightsp++;
		weightschp++;
	}
	biasweight = biasweight + bias * error * learningRate - biasweight * decay * learningRate;
}



double Neuron::getSumOfSquaredWeightVector() {
	double* weightsp = weights;
	unsigned char * maskp = mask;
	double sq = 0;
	for(int i=0;i<nInputs;i++) {
		if (*maskp) {
			const double a = *weightsp;
			sq = sq + a*a;
		}
		maskp++;
		weightsp++;
	}
	sq = sq + biasweight*biasweight;
	return sq;
}



double Neuron::getManhattanNormOfWeightVector() {
	double* weightsp = weights;
	unsigned char * maskp = mask;
	double norm = 0;
	for(int i=0;i<nInputs;i++) {
		if (*maskp) {
			const double a = *weightsp;
			norm = norm + fabs(a);
		}
		maskp++;
		weightsp++;
	}
	norm = norm + fabs(biasweight);
	return norm;
}


double Neuron::getInfinityNormOfWeightVector() {
	double* weightsp = weights;
	unsigned char * maskp = mask;
	double norm = 0;
	for(int i=0;i<nInputs;i++) {
		if (*maskp) {
			const double a = fabs(*weightsp);
			if (a>norm) norm = a;
		}
		maskp++;
		weightsp++;
	}
	const double b = fabs(biasweight);
	if (b > norm) norm = b;
	return norm;
}


double Neuron::getAverageOfWeightVector() {
	double* weightsp = weights;
	unsigned char * maskp = mask;
	double norm = 0;
	long int n = 0;
	for(int i=0;i<nInputs;i++) {
		if (*maskp) {
			const double a = *weightsp;
			norm = norm + a;
			n++;
		}
		maskp++;
		weightsp++;
	}
	norm = norm + fabs(biasweight);
	n++;
	return norm/(double)n;
}


void Neuron::normaliseWeights(double norm) {
	double* weightsp = weights;
	unsigned char * maskp = mask;

	// check for a div by zero
	if (!(fabs(norm) > 0)) return;

	for(int i=0;i<nInputs;i++) {
		if (*maskp) {
			*weightsp = *weightsp / norm;
		}
		maskp++;
		weightsp++;
	}
}
	


void Neuron::doMaxDet() {
	double* inputsp = inputs;
	double* weightsp = weights;
	unsigned char * maskp = mask;
	int maxInp = 0;
	double max = 0;
	maxDet = 1;
	for(int i=0;i<nInputs;i++) {
		const double input = fabs(*inputsp);
		if (*maskp) {
			if (input>max) {
				max = input;
				maxInp = i;
			}
		}
		*weightsp = 0;
		inputsp++;
		maskp++;
		weightsp++;
	}
	weights[maxInp] = 1;
}


void Neuron::initWeights( double _max,  int initBias, WeightInitMethod weightInitMethod) {
	//fprintf(stderr,"Init Weights: max=%f\n",_max);
	double max = _max;
	int nBias = 0;
	if (initBias) nBias++;
	switch (weightInitMethod) {
	case MAX_WEIGHT_RANDOM:
		max = fabs(_max);
		break;
	case MAX_OUTPUT_RANDOM:
		max = fabs(_max) / ((double)(nInputs+nBias));
		break;
	case MAX_OUTPUT_CONST:
		max = _max / (nInputs+nBias);
		break;
	case CONST_WEIGHTS:
		break;
	}
	for(int i=0;i<nInputs;i++) {
		switch (weightInitMethod) {
		case MAX_WEIGHT_RANDOM:
		case MAX_OUTPUT_RANDOM:
			weights[i] = (((double)rand()*2)/((double)RAND_MAX)*max)-max;
			break;
		case CONST_WEIGHTS:
		case MAX_OUTPUT_CONST:
			weights[i] = max;
			break;
		}
		initialWeights[i]=weights[i];
	}
	if (initBias) {
		switch (weightInitMethod) {
		case MAX_WEIGHT_RANDOM:
		case MAX_OUTPUT_RANDOM:
			biasweight = (((double)rand()*2)/((double)RAND_MAX)*max)-max;
			break;
		case CONST_WEIGHTS:
		case MAX_OUTPUT_CONST:
			biasweight = max;
			break;
		}
	}
}


void Neuron::saveInitialWeights() {
	for(int i=0;i<nInputs;i++) {
		initialWeights[i]=weights[i];
	}
}



double Neuron::getMaxWeightValue() {
	double max=-HUGE_VAL;
	for(int i=0;i<nInputs;i++) {
		if (mask[i]) {
			const double w = weights[i];
			if (w>max) max = w;
		}
	}
	if (biasweight > max) max = biasweight;
	return max;
}



double Neuron::getMinWeightValue() {
	double min=HUGE_VAL;
	for(int i=0;i<nInputs;i++) {
		if (mask[i]) {
			const double w = weights[i];
			if (w<min) min = w;
		}
	}
	if (biasweight < min) min = biasweight;
	return min;
}



double Neuron::getWeightDistanceFromInitialWeights() {
	double distance = 0;
	for(int i=0;i<nInputs;i++) {
		if (mask[i]) {
			const double w = weights[i] - initialWeights[i];
			distance += w*w;
		}
	}
	return sqrt(distance);
}



void Neuron::setError(double _error) {
	error = _error;
	assert(!isnan(_error));
}


void Neuron::setMask( int x, int y, unsigned char c) {
	if (x<0) return;
	if (y<0) return;
	if (x>=width) return;
	if (y>=height) return;
	mask[x+y*width] = c;
}

void Neuron::setMask( unsigned char c) {
	for(int i=0;i<nInputs;i++) {
		mask[i] = c;
	}
}

unsigned char Neuron::getMask( int x, int y) {
	if (x<0) return 0;
	if (y<0) return 0;
	if (x>=width) return 0;
	if (y>=height) return 0;
	return mask[x+y*width];
}
