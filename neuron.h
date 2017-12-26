#ifndef __Neuron_H_
#define __Neuron_H_

#include<math.h>
#include<stdio.h>

/**
 * GNU GENERAL PUBLIC LICENSE
 * Version 3, 29 June 2007
 *
 * (C) 2017, Bernd Porr <bernd@glasgowneuro.tech>
 * (C) 2017, Paul Miller <paul@glasgowneuro.tech>
 **/

// bypasses the sigmoid function
// #define LINEAR_OUTPUT

// enables denbug output to sdt out
// #define DEBUG_NEURON

#include "globals.h"
#include "bandpass.h"

class Neuron {

public:

	Neuron(int _nInputs, int _nFilters = 0, double _minT = 0, double _maxT = 0);
	~Neuron();

	// calculate the output of the neuron
	void calcOutput();
	static void* calcOutputThread(void* object) {
		reinterpret_cast<Neuron*>(object)->calcOutput();
		return NULL;
	};

	// does the learning
	void doLearning();
	static void* doLearningThread(void* object) {
		reinterpret_cast<Neuron*>(object)->doLearning();
		return NULL;
	};

	// detects max of an input and switches that weight to 1 and the others to 0
	void doMaxDet();
	static void* doMaxDetThread(void* object) {
		reinterpret_cast<Neuron*>(object)->doMaxDet();
		return NULL;
	};

	// inits the weights
	enum WeightInitMethod { MAX_OUTPUT_RANDOM = 0, MAX_WEIGHT_RANDOM = 1, MAX_OUTPUT_CONST = 2, CONST_WEIGHTS = 3};
	void initWeights(double _max = 1, int initBias = 1, WeightInitMethod _wm = MAX_OUTPUT_RANDOM);

	enum ActivationFunction { LINEAR = 0, TANH = 1, RELU = 2, REMAXLU};
	void setActivationFunction(ActivationFunction _activationFunction) {
		activationFunction = _activationFunction;
	}

	// returns the output of the neuron fed through the derivative of the activation
	double dActivation();

	double getMinWeightValue();
	double getMaxWeightValue();
	double getWeightDistanceFromInitialWeights();

	inline double getOutput() { return output; };
	inline double getSum() { return sum; };
	inline double getWeight( int _index,  int _filter) {
#ifdef RANGE_CHECKS
		if (!((_index>=0)&&(_index<nInputs)&&(_filter>=0)&&(_filter<nFilters))) {
			fprintf(stderr,"BUG! in Neuron::%s, layer=%d, _index=%d, _filter=%d\n",__FUNCTION__,layerIndex,_index,_filter);
			assert(0==1);
		}
#endif
		
		return mask[_index] ? weights[_index][_filter] : 0;
	};
	inline void setWeight( int _index,  double _weight,  int _filter) {
		assert((_index>=0)&&(_index<nInputs)&&(_filter>=0)&&(_filter<nFilters));
		weights[_index][_filter]=_weight;
	};
	void setError(double _error);
	inline double getError() { return error; };
	inline void setInput( int _index,  double _value) {
		assert((_index>=0)&&(_index<nInputs));
		inputs[_index] = _value;
	};
	inline double getInput( int _index) {
		assert((_index>=0)&&(_index<nInputs));
		return inputs[_index];
	};
	inline double getBiasWeight() {return biasweight; };
	inline void setBiasWeight(double _biasweight) { biasweight=_biasweight; };
	inline void setBias( double _bias) { bias=_bias; };
	inline void setLearningRate( double _learningrate) { learningRate = _learningrate; };
	inline void setMomentum( double _momentum) { momentum = _momentum; };
	inline void setUseDerivative( int _useDerivative) { useDerivative = _useDerivative; };
	inline int getNinputs() { return nInputs; };
	inline int getNfilters() { return nFilters; };
	double getAvgWeight(int _input);
	double getAvgWeightChange(int _input);
	double getAvgWeightChange();

	// tells the layer if it's been a 2D array originally to be a convolutional layer
	void setGeometry( int _width,  int _height) {
		assert((_width*_height)==nInputs);
			width = _width;
			height = _height;
	}
	
	// boundary safe manipulation of the mask
	void setMask( int x, int y, unsigned char c);
	
	// boundary safe manipulation of the mask
	void setMask( unsigned char c);
	
	// boundary safe return of the mask
	unsigned char getMask( int x, int y);

	// mask in linear form
	unsigned char getMask( int index) { return mask[index]; };

	// calculates the Eucledian length of the weight vector
	double getEuclideanNormOfWeightVector();

	// calculates the Manhattan length of the weight vector
	double getManhattanNormOfWeightVector();

	// calculates the average of the weight values
	double getAverageOfWeightVector();

	// normalises weights
	void normaliseWeights();

	void saveInitialWeights();

	// enables debug output
	void setDebugInfo(int _layerIndex, int _neuronIndex) {
		layerIndex = _layerIndex;
		neuronIndex = _neuronIndex;
	}

	inline void setStep(long int _step) {
		step = _step;
	}

private:
	void calcFilterbankOutput();
	void calcOutputWithoutFilterbank();

	void doLearningWithFilterbank();
	void doLearningWithoutFilterbank();
	
private:
	int nInputs;
	unsigned char* mask = 0;
	int nFilters;
	double** weights = 0;
	double** initialWeights = 0;
	double** weightChange = 0;
	double biasweight = 0;
	double biasweightChange = 0;
	double bias = 0;
	Bandpass ***bandpass = 0;
	double* inputs = 0;
	double output = 0;
	double sum = 0;
	double error = 0;
	double internal_error = 0;
	double learningRate = 0;
	double learningRateFactor = 1;
	double momentum = 0;
	double minT,maxT;
	double dampingCoeff = 0.51;
	int useDerivative = 0;
	double oldError = 0;
	int width = 0;
	int height = 0;
	int maxDet = 0;
	int layerIndex = 0;
	int neuronIndex = 0;
	long int step = 0;
	ActivationFunction activationFunction = TANH;
};

#endif
