#ifndef __Neuron_H_
#define __Neuron_H_

#include<math.h>

/**
 * GNU GENERAL PUBLIC LICENSE
 * Version 3, 29 June 2007
 *
 * (C) 2017, Bernd Porr <mail@berndporr.me.uk>, <bernd@glasgowneuro.tech>
 * (C) 2017, Paul Miller <nlholdem@hotmail.com>
 **/

// bypasses the sigmoid function
// #define LINEAR_OUTPUT

// enables denbug output to sdt out
// #define DEBUG_NEURON

#include "bandpass.h"

class Neuron {

public:

	Neuron(int _nInputs, int _nFilters = 0, double _minT = 0, double _maxT = 0);
	~Neuron();
	void calcOutput();
	static void* calcOutputThread(void* object) {
		reinterpret_cast<Neuron*>(object)->calcOutput();
	};
	void doLearning();
	static void* doLearningThread(void* object) {
		reinterpret_cast<Neuron*>(object)->doLearning();
	};
	void doMaxDet();
	static void* doMaxDetThread(void* object) {
		reinterpret_cast<Neuron*>(object)->doMaxDet();
	};	
	void initWeights(double _max, int initBias);
	inline double getOutput() { return output; };
	inline double getSum() { return sum; };
	inline double getWeight( int _index,  int _filter = 0) {
		assert((_index>=0)&&(_index<nInputs)&&(_filter>=0)&&(_filter<nFilters));
		return mask[_index] ? weights[_index][_filter] : 0;
	};
	inline void setWeight( int _index,  double _weight,  int _filter = 0) { weights[_index][_filter]=_weight; };
	void setError(double _error);
	inline double getError() { return error; };
	inline void setInput( int _index,  double _value) { inputs[_index] = _value; };
	inline double getInput( int _index) { return inputs[_index]; };
	inline void setBias( double _bias) { bias=_bias; };
	inline void setLearningRate( double _learningrate) { learningRate = _learningrate; };
	inline void setUseDerivative( int _useDerivative) { useDerivative = _useDerivative; };
	inline int getNinputs() { return nInputs; };
	double getAvgWeight(int _input);

	// tells the layer if it's been a 2D array originally
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

	// normalises weights
	void normaliseWeights();

	// enables debug output
	void enableDebugging(int _layerIndex) {
		layerIndex = _layerIndex;
		debugOutput = 1;
	}

private:
	int nInputs;
	unsigned char* mask = 0;
	int nFilters;
	double** weights = 0;
	double biasweight = 0;
	double bias = 0;
	Bandpass ***bandpass = 0;
	double* inputs = 0;
	double output = 0;
	double sum = 0;
	double error = 0;
	double learningRate = 0;
	double minT,maxT;
	double dampingCoeff = 0.51;
	int useDerivative = 1;
	double oldError = 0;
	int width = 0;
	int height = 0;
	int maxDet = 0;
	int layerIndex = 0;
	int debugOutput = 0;
};

#endif
