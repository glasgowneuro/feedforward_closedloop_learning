#include "neuron.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

Neuron::Neuron(int _nInputs, int _nFilters, double _minT, double _maxT) {
	nInputs = _nInputs;
	nFilters = _nFilters;
	minT = _minT;
	maxT = _maxT;
	dampingCoeff = 0.51;
	biasweight = 0;
	bias = 0;
#ifdef DEBUG_NEURON
	fprintf(stderr,"creating %d weights: ",nInputs);
#endif
	weights = new double*[nInputs];
#ifdef DEBUG_NEURON
	fprintf(stderr,"done\n");
#endif
	useDerivative = 0;
	oldError = 0;
	if (nFilters>0) {
		bandpass = new Bandpass**[nInputs];
	} else {
		bandpass = NULL;
		nFilters = 1;
	}
	for(int i=0;i<nInputs;i++) {
		weights[i] = new double[nFilters];
		if (bandpass != NULL) {
			bandpass[i] = new Bandpass*[nFilters];
			double fs = 1;
			double fmin = fs/maxT;
			double fmax = fs/minT;
			double df = (fmax-fmin)/((double)nFilters);
			double f = fmin;
#ifdef DEBUG_BP
			fprintf(stderr,"fmin=%f,fmax=%f,df=%f\n",fmin,fmax,df);
#endif
			for(int j=0;j<_nFilters;j++) {
				bandpass[i][j] = new Bandpass();
#ifdef DEBUG_BP
				fprintf(stderr,"bandpass[%d][%d]->setup(2,%f,%f)\n",
					i,j,fs,f);
#endif
				bandpass[i][j]->setParameters(f,dampingCoeff);
				f = f + df;
				for(int k=0;k<maxT;k++) {
					float a = 0;
					if (k==minT) {
						a = 1;
					}
					float b = bandpass[i][j]->filter(a);
					assert(b != NAN);
					assert(b != INFINITY);
				}
			}
		}
	}
	inputs = new double[nInputs];
	sum = 0;
	output = 0;
	bias = 0;
	error = 0;
	learningRate = 0;
	for(int i=0;i<nInputs;i++) {
		for(int j=0;j<nFilters;j++) {
			weights[i][j] = 0;
		}
		inputs[i] = 0;
	}	
}

Neuron::~Neuron() {
	delete [] weights;
	delete [] inputs;
}


void Neuron::calcOutput() {
	sum = 0;

	if (bandpass) {
		double** weightsp1 = weights;
		Bandpass*** bandpassp1 = bandpass;
		Bandpass** bandpassp2;
		double* inputp = inputs;
		for(int i=0;i<nInputs;i++) {
			double input = *inputp;
			assert(inputs[i] == input);
			double* weightsp2 = *weightsp1;
				bandpassp2 = *bandpassp1;
				bandpassp1++;
			weightsp1++;
			inputp++;
			for(int j = 0;j<nFilters;j++) {
#ifdef DEBUG_NEURON
				assert(weights[i][j] == (*weightsp2));
#endif
				sum = sum + (*weightsp2) * (*bandpassp2)->filter(input);
				bandpassp2++;
				weightsp2++;
#ifdef DEBUG_NEURON
				if (isnan(sum) || isnan(weights[i][j]) || isnan(inputs[i])) {
					printf("calcOutput: %f, %f, %f, %d, %d\n",sum,weights[i][j],inputs[i],i,j);
					exit(EXIT_FAILURE);
				}
#endif
			}
		}
	} else {
		double** weightsp1 = weights;
		double* inputp = inputs;
		for(int i=0;i<nInputs;i++) {
			double input = *inputp;
			assert(inputs[i] == input);
			double* weightsp2 = *weightsp1;
			weightsp1++;
			inputp++;
			for(int j = 0;j<nFilters;j++) {
#ifdef DEBUG_NEURON
				assert(weights[i][j] == (*weightsp2));
#endif
				sum = sum + (*weightsp2) * input;
				weightsp2++;
#ifdef DEBUG_NEURON
				if (isnan(sum) || isnan(weights[i][j]) || isnan(inputs[i])) {
					printf("calcOutput: %f, %f, %f, %d, %d\n",sum,weights[i][j],inputs[i],i,j);
					exit(EXIT_FAILURE);
				}
#endif
			}
		}
		
	}

	sum = sum + biasweight * bias;
	
#ifdef LINEAR_OUTPUT
	output = sum;
#else
	output = tanh(sum);
#endif
}


void Neuron::doLearning() {
	double* inputsp = inputs;
	double** weightsp1 = weights;
	for(int i=0;i<nInputs;i++) {
		double input = *inputsp;
		inputsp++;
		double* weightsp2 = *weightsp1;
		weightsp1++;
		for(int j=0;j<nFilters;j++) {
			*weightsp2 = *weightsp2 + input * error * learningRate;
			weightsp2++;
#ifdef DEBUG_NEURON
			if (isnan(weights[i][j]) || isnan(inputs[i]) || isnan (error)) {
				printf("Neuron::doLearning: %f,%f,%f\n",weights[i][j],inputs[i],error);
				exit(EXIT_FAILURE);
			}
#endif
		}
	}
	biasweight = biasweight + bias * error * learningRate;
	//printf("b=%e\n",biasweight);
}


void Neuron::initWeights(double _max, int initBias) {
	for(int i=0;i<nInputs;i++) {
		for(int j=0;j<nFilters;j++) {
			if (_max>0) {
				weights[i][j] = (((double)random())/((double)RAND_MAX)*_max);
			} else {
				weights[i][j] = 0;
			}
		}
	}
	if (initBias) {
		biasweight=(((double)random())/((double)RAND_MAX)*_max);
	}
}

double Neuron::getAvgWeight(int _input) {
	double w = 0;
	for(int j=0;j<nFilters;j++) {
		w += weights[_input][j];
	}
	w += biasweight;
	return w;
}



void Neuron::setError(double _error) {
#ifdef DEBUG_NEURON
	if (isnan(_error)) {
			printf(" Neuron::setError: error=%f\n",_error);
			exit(1);
	}
#endif
	if (useDerivative) {
		error = _error - oldError;
		oldError = _error;
	} else {
		error = _error;
	}
}
