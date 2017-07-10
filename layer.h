#ifndef __Layer_H_
#define __Layer_H_

/**
 * GNU GENERAL PUBLIC LICENSE
 * Version 3, 29 June 2007
 *
 * (C) 2017, Bernd Porr <mail@berndporr.me.uk>, <bernd@glasgowneuro.tech>
 * (C) 2017, Paul Miller <nlholdem@hotmail.com>
 **/

#include "neuron.h"

class Layer {
	
public:
	Layer(int _nNeurons, int _nInputs, int _nFilters = 0, double _minT = 0, double _maxT = 0);
	~Layer();

	void calcOutputs();
	void doLearning();

	// sets the global error for all neurons
	void setError(const double _error);

	// sets the error individually
	void setError(const int i, const double _error);

	// sets all errors from an input array
	void setErrors(const double *_errors);

	// retrieves the error
	double getError(const int i);

	// sets the global error for all neurons
	void setBias(const double _bias);

	// sets if we use the derivative
	void setUseDerivative(const int useIt);

	// this is used to copy the output from the previous
	// layer into this input layer or to the sensor inputs
	void setInput(const int inputIndex, const double input);

	// sets all inputs from an input array
	void setInputs(const double * _inputs, double min=0, double max=0);

	// sets the learning rate of all neurons
	void setLearningRate(const double _learningRate);

	// inits weights with a random value between -_max and max
	void initWeights(const double _max, const int initBiasWeight = 0);

	// gets the outpuut of one neuron
	inline double getOutput(const int index) {
		return neurons[index]->getOutput();
	}

	// gets a pointer to one neuron
	inline Neuron* getNeuron(const int index) {
		return neurons[index];
	}

	// number of neurons
	inline int getNneurons() { return nNeurons;}

	// number of inputs
	inline int getNinputs() { return nInputs;}

	void setConvolution(const int width, const int height);

	void setMaxDetLayer(int _m) { maxDetLayer = _m; };

	void setNormaliseWeights(int _normaliseWeights) { normaliseWeights = _normaliseWeights;};

private:
	int nNeurons;
	int nInputs;
	int nFilters;
	Neuron** neurons = 0;
	double minT;
	double maxT;
	int maxDetLayer = 0;
	int normaliseWeights = 0;
};

#endif
