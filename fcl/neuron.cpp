#include "neuron.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

/**
 * GNU GENERAL PUBLIC LICENSE
 * Version 3, 29 June 2007
 *
 * (C) 2017, Bernd Porr <bernd@glasgowneuro.tech>
 * (C) 2017, Paul Miller <paul@glasgowneuro.tech>
 **/

#define SUM_ERR_THRES 1000

Neuron::Neuron(int _nInputs, int _nFilters, double _minT, double _maxT) {
	nInputs = _nInputs;
	nFilters = _nFilters;
	minT = _minT;
	maxT = _maxT;

	mask = new unsigned char[nInputs];
	weights = new double*[nInputs];
	initialWeights = new double*[nInputs];
	weightChange = new double*[nInputs];

	if (nFilters>0) {
		bandpass = new Bandpass**[nInputs];
	} else {
		// acts as a flag that there are no filters
		bandpass = NULL;
		nFilters = 1;
	}

	for(int i=0;i<nInputs;i++) {
		weights[i] = new double[nFilters];
		initialWeights[i] = new double[nFilters];
		weightChange[i] = new double[nFilters];
		if (bandpass != NULL) {
			bandpass[i] = new Bandpass*[nFilters];
			 double fs = 1;
			 double fmin = fs/maxT;
			 double fmax = fs/minT;
			 double df = (fmax-fmin)/((double)(nFilters-1));
			 double f = fmin;
#ifdef DEBUG_BP
			fprintf(stderr,"bandpass: fmin=%f,fmax=%f,df=%f\n",fmin,fmax,df);
#endif
			for(int j=0;j<_nFilters;j++) {
				bandpass[i][j] = new Bandpass();
#ifdef DEBUG_BP
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
	inputs = new double[nInputs];
	sum = 0;
	output = 0;
	error = 0;
	internal_error = 0;
	learningRate = 0;
	for(int i=0;i<nInputs;i++) {
		for(int j=0;j<nFilters;j++) {
			weights[i][j] = 0;
			initialWeights[i][j] = 0;
			weightChange[i][j] = 0;
		}
		inputs[i] = 0;
		mask[i] = 1;
	}	
}

Neuron::~Neuron() {
	for(int i=0;i<nInputs;i++) {
		delete[] weights[i];
		delete[] initialWeights[i];
		delete[] weightChange[i];
	}
	delete [] weights;
	delete [] initialWeights;
	delete [] weightChange;
	delete [] inputs;
	delete [] mask;
}



void Neuron::calcFilterbankOutput() {
	double** weightsp1 = weights;
	Bandpass*** bandpassp1 = bandpass;
	double* inputp = inputs;
	unsigned char * maskp = mask;

	// global variable
	sum = 0;
	
	for(int i=0;i<nInputs;i++) {
		if (*maskp) {
			double input = *inputp;
			assert(inputs[i] == input);
			double* weightsp2 = *weightsp1;
			Bandpass** bandpassp2 = *bandpassp1;
			for(int j = 0;j<nFilters;j++) {
				assert(weights[i][j] == (*weightsp2));
				sum = sum + (*weightsp2) * (*bandpassp2)->filter(input);
#ifdef RANGE_CHECKS
				if (isnan(sum) || isnan(weights[i][j]) || isnan(inputs[i]) || (sum>SUM_ERR_THRES)) {
					fprintf(stderr,"Out of range Neuron::%s step=%ld, L=%d, N=%d, sum=%f, weights=%f, inputs=%f, bandpass=%f, i=%d, j=%d\n",
						__func__,step,layerIndex,neuronIndex,sum,weights[i][j],inputs[i],(*bandpassp2)->getOutput(),i,j);
				}
#endif
				bandpassp2++;
				weightsp2++;
			}
		}
		maskp++;
		bandpassp1++;
		weightsp1++;
		inputp++;
	}
}



void Neuron::calcOutputWithoutFilterbank() {
	double** weightsp1 = weights;
	double* inputp = inputs;
	unsigned char * maskp = mask;

	// global variable
	sum = 0;

	for(int i=0;i<nInputs;i++) {
		if (*maskp) {
			double input = *inputp;
			assert(inputs[i] == input);
			double* weightsp2 = *weightsp1;
			for(int j = 0;j<nFilters;j++) {
				assert(weights[i][j] == (*weightsp2));
				sum = sum + (*weightsp2) * input;
				weightsp2++;
#ifdef RANGE_CHECKS
				if (isnan(sum) || isnan(weights[i][j]) || isnan(inputs[i]) || (fabs(sum)>SUM_ERR_THRES)) {
					fprintf(stderr,"Out of range Neuron::%s step=%ld, L=%d, N=%d, %f, %f, %f, %d, %d\n",
						__func__,step,layerIndex,neuronIndex,sum,weights[i][j],inputs[i],i,j);
				}
#endif
			}
		}
		weightsp1++;
		inputp++;
		maskp++;
	}
}



void Neuron::calcOutput() {

	if (bandpass) {
		calcFilterbankOutput();
	} else {
		calcOutputWithoutFilterbank();
	}

	sum = sum + biasweight * bias;

#ifdef RANGE_CHECKS
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
		fprintf(stderr,"BUG: undefined activation function in Neuron::%s\n",__FUNCTION__);
		assert(1==0);
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
		fprintf(stderr,"BUG: undefined activation function in Neuron::%s\n",__FUNCTION__);
		assert(1==0);
		return 0;
	}
}



void Neuron::doLearning() {
	if (bandpass) {
		doLearningWithFilterbank();
	} else {
		doLearningWithoutFilterbank();
	}
}

void Neuron::doLearningWithFilterbank() {
	double** weightsp1 = weights;
	double** weightschp1 = weightChange;
	unsigned char * maskp = mask;
	Bandpass*** bandpassp1 = bandpass;
	maxDet = 0;
	for(int i=0;i<nInputs;i++) {
		Bandpass** bandpassp2 = *bandpassp1;
		double* weightsp2 = *weightsp1;
		double* weightschp2 = *weightschp1;
		if (*maskp) {
			for(int j=0;j<nFilters;j++) {
				*weightschp2 = momentum * (*weightschp2) +
					(*bandpassp2)->getOutput() * internal_error * learningRate * learningRateFactor -
					(*weightsp2) * decay * learningRate * fabs(internal_error);
				*weightsp2 = *weightsp2 + *weightschp2;
#ifdef RANGE_CHECKS				
				if (*weightsp2 > 10000) printf("Neuron::%s, step=%ld, L=%d,N=%d (%d,%d,%e,%e,%e,%e)\n",
							       __func__,
							       step,layerIndex,neuronIndex,
							       i,j,*weightsp2,(*bandpassp2)->getOutput(),internal_error,learningRate);
#endif
				weightsp2++;
				weightschp2++;
				bandpassp2++;
#ifdef RANGE_CHECKS
				if (isnan(weights[i][j]) || isnan(inputs[i]) || isnan (internal_error)) {
					printf("Neuron::%s: step=%ld, L=%d, %f,%f,%f\n",
					       __func__,
					       step,layerIndex,
					       weights[i][j],inputs[i],internal_error);
					exit(EXIT_FAILURE);
				}
#endif
			}
		}
		bandpassp1++;
		maskp++;
		weightsp1++;
		weightschp1++;
	}
//	printf("\n");
	biasweight = biasweight + bias * internal_error * learningRate - biasweight * decay * learningRate;
}


void Neuron::doLearningWithoutFilterbank() {
	double* inputsp = inputs;
	double** weightsp1 = weights;
	double** weightschp1 = weightChange;
	unsigned char * maskp = mask;
	maxDet = 0;
	for(int i=0;i<nInputs;i++) {
		if (*maskp) {
			double input = *inputsp;
			double* weightsp2 = *weightsp1;
			double* weightschp2 = *weightschp1;
			for(int j=0;j<nFilters;j++) {
				*weightschp2 = momentum * (*weightschp2) +
					input * internal_error * learningRate * learningRateFactor -
					(*weightsp2) * decay * learningRate * fabs(internal_error);
				*weightsp2 = *weightsp2 + *weightschp2;
				weightsp2++;
				weightschp2++;
#ifdef RANGE_CHECKS
				if (isnan(sum) || isnan(weights[i][j]) || isnan(inputs[i]) || (fabs(sum)>SUM_ERR_THRES)) {
					fprintf(stderr,"Out of range Neuron::%s step=%ld, L=%d, N=%d, %f, %f, %f, %d, %d\n",
						__func__,step,layerIndex,neuronIndex,sum,weights[i][j],inputs[i],i,j);
				}
#endif
			}
		}
		inputsp++;
		maskp++;
		weightsp1++;
		weightschp1++;
	}
//	printf("\n");
	biasweight = biasweight + bias * internal_error * learningRate - biasweight * decay * learningRate;
}



double Neuron::getSumOfSquaredWeightVector() {
	double** weightsp1 = weights;
	unsigned char * maskp = mask;
	double sq = 0;
	for(int i=0;i<nInputs;i++) {
		if (*maskp) {
			double* weightsp2 = *weightsp1;
			for(int j=0;j<nFilters;j++) {
				double a = *weightsp2;
				sq = sq + a*a;
				weightsp2++;
			}
		}
		maskp++;
		weightsp1++;
	}
	sq = sq + biasweight*biasweight;
	return sq;
}



double Neuron::getManhattanNormOfWeightVector() {
	double** weightsp1 = weights;
	unsigned char * maskp = mask;
	double norm = 0;
	for(int i=0;i<nInputs;i++) {
		if (*maskp) {
			double* weightsp2 = *weightsp1;
			for(int j=0;j<nFilters;j++) {
				double a = *weightsp2;
				norm = norm + fabs(a);
				weightsp2++;
			}
		}
		maskp++;
		weightsp1++;
	}
	norm = norm + fabs(biasweight);
	return norm;
}


double Neuron::getInfinityNormOfWeightVector() {
	double** weightsp1 = weights;
	unsigned char * maskp = mask;
	double norm = 0;
	for(int i=0;i<nInputs;i++) {
		if (*maskp) {
			double* weightsp2 = *weightsp1;
			for(int j=0;j<nFilters;j++) {
				double a = fabs(*weightsp2);
				if (a>norm) norm = a;
				weightsp2++;
			}
		}
		maskp++;
		weightsp1++;
	}
	norm = norm + fabs(biasweight);
	return norm;
}


double Neuron::getAverageOfWeightVector() {
	double** weightsp1 = weights;
	unsigned char * maskp = mask;
	double norm = 0;
	long int n = 0;
	for(int i=0;i<nInputs;i++) {
		if (*maskp) {
			double* weightsp2 = *weightsp1;
			for(int j=0;j<nFilters;j++) {
				double a = *weightsp2;
				norm = norm + a;
				weightsp2++;
				n++;
			}
		}
		maskp++;
		weightsp1++;
	}
	norm = norm + fabs(biasweight);
	n++;
	return norm/(double)n;
}


void Neuron::normaliseWeights(double norm) {
	double** weightsp1 = weights;
	unsigned char * maskp = mask;
	
	if (fabs(norm) > 0) {
		weightsp1 = weights;
		maskp = mask;
		for(int i=0;i<nInputs;i++) {
			if (*maskp) {
				double* weightsp2 = *weightsp1;
				for(int j=0;j<nFilters;j++) {
					*weightsp2 = *weightsp2 / norm;
#ifdef RANGE_CHECKS
					if (fabs(*weightsp2) > 1000)
						fprintf(stderr,"Neuron::%s, step=%ld, L=%d, N=%d, %d,%d,weight=%e,norm=%e\n",__func__,step,layerIndex,neuronIndex,i,j,*weightsp2,norm);
#endif
					weightsp2++;
				}
			}
			maskp++;
			weightsp1++;
		}
	}
}
	


void Neuron::doMaxDet() {
	double* inputsp = inputs;
	double** weightsp1 = weights;
	unsigned char * maskp = mask;
	int maxInp = 0;
	double max = 0;
	maxDet = 1;
	for(int i=0;i<nInputs;i++) {
		double input = fabs(*inputsp);
		double* weightsp2 = *weightsp1;
		if (*maskp) {
			if (input>max) {
				max = fabs(input);
				maxInp = i;
			}
		}
		for(int j=0;j<nFilters;j++) {
			*weightsp2 = 0;
			weightsp2++;
		}
		inputsp++;
		maskp++;
		weightsp1++;
	}
	weights[maxInp][0] = 1;
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
		max = fabs(_max) / ((double)(nInputs*nFilters+nBias));
		break;
	case MAX_OUTPUT_CONST:
		max = _max / (nInputs*nFilters+nBias);
		break;
	case CONST_WEIGHTS:
		break;
	}
	for(int i=0;i<nInputs;i++) {
		for(int j=0;j<nFilters;j++) {
			switch (weightInitMethod) {
			case MAX_WEIGHT_RANDOM:
			case MAX_OUTPUT_RANDOM:
				weights[i][j] = (((double)rand()*2)/((double)RAND_MAX)*max)-max;
				//fprintf(stderr,"Init Weights: weight(%d,%d)=%f\n",i,j,weights[i][j]);
				break;
			case CONST_WEIGHTS:
			case MAX_OUTPUT_CONST:
				weights[i][j] = max;
				break;
			}
			initialWeights[i][j]=weights[i][j];
		}
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
		for(int j=0;j<nFilters;j++) {
			initialWeights[i][j]=weights[i][j];
		}
	}
}



double Neuron::getMaxWeightValue() {
	double max=-HUGE_VAL;
	for(int i=0;i<nInputs;i++) {
		if (mask[i]) {
			for(int j=0;j<nFilters;j++) {
				double w = weights[i][j];
				if (w>max) w = max;
			}
		}
	}
	if (biasweight > max) max = biasweight;
	return max;
}



double Neuron::getMinWeightValue() {
	double min=HUGE_VAL;
	for(int i=0;i<nInputs;i++) {
		if (mask[i]) {
			for(int j=0;j<nFilters;j++) {
				double w = weights[i][j];
				if (w<min) w = min;
			}
		}
	}
	if (biasweight < min) min = biasweight;
	return min;
}



double Neuron::getWeightDistanceFromInitialWeights() {
	double distance = 0;
	for(int i=0;i<nInputs;i++) {
		if (mask[i]) {
			for(int j=0;j<nFilters;j++) {
				double w = weights[i][j] - initialWeights[i][j];
				distance += w*w;
			}
		}
	}
	return sqrt(distance);
}



double Neuron::getAvgWeight( int _input) {
	if (!mask[_input]) return 0;
	int n=0;
	double w=0;
	for(int j=0;j<nFilters;j++) {
		w += weights[_input][j];
		n++;
	}
	return w/((double)n);
}

double Neuron::getAvgWeightChange( int _input) {
	if (!mask[_input]) return 0;
	int n=0;
	double wch=0;
	for(int j=0;j<nFilters;j++) {
		wch += weightChange[_input][j];
		n++;
	}
//	wch+= biasweightChange;
//	n++;
	return wch/((double)n);
}

double Neuron::getAvgWeightChange() {
	double wch=0;
	int n=0;
	for(int i=0;i<nInputs;i++) {
		if (mask[i]) {
			wch += getAvgWeightChange(i);
			n++;
		}
	}
	return wch/((double)n);
}



void Neuron::setError( double _error) {
	error = _error;
#ifdef DEBUG_NEURON
	if (isnan(_error)) {
			printf(" Neuron::setError: error=%f\n",_error);
			exit(1);
	}
#endif
	if (useDerivative) {
		internal_error = _error - oldError;
		oldError = _error;
	} else {
		internal_error = _error;
	}
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
