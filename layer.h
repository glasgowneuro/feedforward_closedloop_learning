#ifndef __Layer_H_
#define __Layer_H_

/**
 * GNU GENERAL PUBLIC LICENSE
 * Version 3, 29 June 2007
 *
 * (C) 2017, Bernd Porr <bernd@glasgowneuro.tech>
 * (C) 2017, Paul Miller <paul@glasgowneuro.tech>
 **/

#include "globals.h"
#include "neuron.h"

class Layer {
	
public:
	Layer(int _nNeurons, int _nInputs, int _nFilters = 0, double _minT = 0, double _maxT = 0);
	~Layer();

	
	void calcOutputs();
	void doLearning();

	// sets the global error for all neurons
	void setError( double _error);

	// sets the error individually
	void setError( int i,  double _error);

	// sets all errors from an input array
	void setErrors( double *_errors);

	// retrieves the error
	double getError( int i);

	// sets the global error for all neurons
	void setBias( double _bias);

	// sets if we use the derivative
	void setUseDerivative( int useIt);

	// this is used to copy the output from the previous
	// layer into this input layer or to the sensor inputs
	void setInput( int inputIndex,  double input);

	// sets all inputs from an input array
	void setInputs( double * _inputs);

	// sets the learning rate of all neurons
	void setLearningRate( double _learningRate);

	void setActivationFunction(Neuron::ActivationFunction _activationFunction);

	// set the momentum of all neurons in this layer
	void setMomentum( double _momentum);

	// inits weights with a random value between -_max and max
	void initWeights( double _max = 1,
			  int initBiasWeight = 1,
			  Neuron::WeightInitMethod weightInitMethod = Neuron::MAX_OUTPUT_RANDOM);
	
	// gets the outpuut of one neuron
	inline double getOutput( int index) {
		return neurons[index]->getOutput();
	}

	// gets a pointer to one neuron
	inline Neuron* getNeuron( int index) {
		assert(index < nNeurons);
		return neurons[index];
	}

	// number of neurons
	inline int getNneurons() { return nNeurons;}

	// number of inputs
	inline int getNinputs() { return nInputs;}

	void setConvolution( int width,  int height);

	void setMaxDetLayer(int _m) { maxDetLayer = _m; };

	void setNormaliseWeights(int _normaliseWeights);

	void setDebugInfo(int layerIndex);

	void setStep(long int step);

	double getWeightDistanceFromInitialWeights();
	
private:

	enum InputNormMethod { INPUT_NORM_NONE=0, INPUT_NORM_ZEROMEAN_AUTO = 1, INPUT_NORM_ZEROMEAN_MANUAL = 2 };

	void setInputsNormalised2zeroMean( double* inputs, int doCalc = 1 );

	void setInputsWithoutNormalisation( double* inputs );
	
	int nNeurons;
	int nInputs;
	int nFilters;
	Neuron** neurons = 0;
	double minT;
	double maxT;
	int maxDetLayer = 0;
	int normaliseWeights = 0;
	int debugOutput = 0;
	// for debugging output
	int layerIndex = 0;
	long int step = 0;
};

#endif
