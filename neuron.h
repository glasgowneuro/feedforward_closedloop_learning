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
	void initWeights(double _max, int initBias);
	inline double getOutput() { return output; };
	inline double getSum() { return sum; };
	inline double getWeight(const int _index, const int _filter = 0) { return mask[_index] ? weights[_index][_filter] : 0; };
	inline void setWeight(const int _index, const double _weight, const int _filter = 0) { weights[_index][_filter]=_weight; };
	void setError(double _error);
	inline double getError() { return error; };
	inline void setInput(const int _index, const double _value) { inputs[_index] = _value; };
	inline double getInput(const int _index) { return inputs[_index]; };
	inline void setBias(const double _bias) { bias=_bias; };
	inline void setLearningRate(const double _learningrate) { learningRate = _learningrate; };
	inline void setUseDerivative(const int _useDerivative) { useDerivative = _useDerivative; };
	inline int getNinputs() { return nInputs; };
	double getAvgWeight(int _input);

	// tells the layer if it's been a 2D array originally
	void setGeometry(const int _width, const int _height) {
		assert((_width*_height)==nInputs);
			width = _width;
			height = _height;
	}
	
	// boundary safe manipulation of the mask
	void setMask(const int x,const int y,const unsigned char c);
	
	// boundary safe manipulation of the mask
	void setMask(const unsigned char c);
	
	// boundary safe return of the mask
	unsigned char getMask(const int x,const int y);

	// mask in linear form
	unsigned char getMask(const int index) { return mask[index]; };

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
};

#endif
