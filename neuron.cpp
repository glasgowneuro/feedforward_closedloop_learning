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
	mask = new unsigned char[nInputs];
	
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
		mask[i] = 1;
	}	
}

Neuron::~Neuron() {
	for(int i=0;i<nInputs;i++) {
		delete[] weights[i];
	}
	delete [] weights;
	delete [] inputs;
	delete [] mask;
}


void Neuron::calcOutput() {
	sum = 0;

	if (bandpass) {
		double** weightsp1 = weights;
		Bandpass*** bandpassp1 = bandpass;
		double* inputp = inputs;
		unsigned char * maskp = mask;
		for(int i=0;i<nInputs;i++) {
			if (*maskp) {
				double input = *inputp;
#ifdef DEBUG_NEURON
				assert(inputs[i] == input);
#endif
				double* weightsp2 = *weightsp1;
				Bandpass** bandpassp2 = *bandpassp1;
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
			maskp++;
			bandpassp1++;
			weightsp1++;
			inputp++;
		}
	} else {
		double** weightsp1 = weights;
		double* inputp = inputs;
		unsigned char * maskp = mask;
		for(int i=0;i<nInputs;i++) {
			if (*maskp) {
				double input = *inputp;
#ifdef DEBUG_NEURON
				
				assert(inputs[i] == input);
#endif
				double* weightsp2 = *weightsp1;
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
			weightsp1++;
			inputp++;
			maskp++;
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
	unsigned char * maskp = mask;		
	for(int i=0;i<nInputs;i++) {
		if (*maskp) {
			double input = *inputsp;
			double* weightsp2 = *weightsp1;
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
		inputsp++;
		maskp++;
		weightsp1++;
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
	int n=0;
	double w=0;
	if (_input < 0) {
		for(int i=0;i<nInputs;i++) {
			for(int j=0;j<nFilters;j++) {
				w += weights[i][j];
				n++;
			}
		}
	} else {
		for(int j=0;j<nFilters;j++) {
			w += weights[_input][j];
			n++;
		}
	}
	w+= biasweight;
	n++;
	return w/((double)n);
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


void Neuron::setMask(int x,int y,unsigned char c) {
	if (x<0) return;
	if (y<0) return;
	if (x>=width) return;
	if (y>=height) return;
	mask[x+y*width] = c;
}

void Neuron::setMask(unsigned char c) {
	for(int i=0;i<nInputs;i++) {
		mask[i] = c;
	}
}

unsigned char Neuron::getMask(int x,int y) {
	if (x<0) return 0;
	if (y<0) return 0;
	if (x>=width) return 0;
	if (y>=height) return 0;
	return mask[x+y*width];
}
