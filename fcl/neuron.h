#ifndef __Neuron_H_
#define __Neuron_H_

#include<math.h>
#include<stdio.h>
#include<assert.h>

// bypasses the sigmoid function
// #define LINEAR_OUTPUT

// enables denbug output to sdt out
// #define DEBUG_NEURON

#include "globals.h"

/** 
 * Neuron which calculates the output and performs learning
 **/
class FCLNeuron {

public:

	/** Constructor
         * \param _nInputs Number of inputs to the Neuron
         **/
	FCLNeuron(int _nInputs);

        /** Destructor
	 * Tidies up any memory allocations
         **/
	~FCLNeuron();

	/** Calculate the output of the neuron
         * This runs the filters, activation functions, sum it all up.
         **/
	void calcOutput();

	/** Wrapper for thread callback for output calc
         **/
	static void* calcOutputThread(void* object) {
		reinterpret_cast<FCLNeuron*>(object)->calcOutput();
		return NULL;
	};

	/** Performs the learning
         * Performs ICO learning in the neuron: pre * error
         **/
	void doLearning();

	/** Wrapper for thread callback for learning
         **/
	static void* doLearningThread(void* object) {
		reinterpret_cast<FCLNeuron*>(object)->doLearning();
		return NULL;
	};

	/** Detects max of an input 
         * Switches the highest weight to 1 and the others to 0
         **/
	void doMaxDet();

	/** Wrapper for thread callback for maxdet
         **/
	static void* doMaxDetThread(void* object) {
		reinterpret_cast<FCLNeuron*>(object)->doMaxDet();
		return NULL;
	};

	/** Constants how to init the weights in the neuron
         **/
	enum WeightInitMethod { MAX_OUTPUT_RANDOM = 0, MAX_WEIGHT_RANDOM = 1, MAX_OUTPUT_CONST = 2, CONST_WEIGHTS = 3};

	/** Inits the weights in the neuron
         * \param _max Maximum value of the weights.
         * \param initBias If one also the bias weight is initialised.
         * \param _wm Method how to init the weights as defined by WeightInitMethod.
         **/
	void initWeights(double _max = 1, int initBias = 1, WeightInitMethod _wm = MAX_OUTPUT_RANDOM);

	/** Activation functions on offer
         * LINEAR: linear unit, 
         * TANH: tangens hyperbolicus, 
         * RELU: linear rectifier, 
         * REMAXLU: as RELU but limits to one.
         **/
	enum ActivationFunction { LINEAR = 0, TANH = 1, RELU = 2, REMAXLU = 3, TANHLIMIT = 4};

	/** Sets the activation function
         * \param _activationFunction Sets the activiation function according to ActivationFunction.
         **/
	void setActivationFunction(ActivationFunction _activationFunction) {
		activationFunction = _activationFunction;
	}

	/** Returns the output of the neuron fed through the derivative of the activation
         * \return Result
         **/
	double dActivation();

	/** Minimum weight value
         * \return The minimum weight value in this neuron
         **/
	double getMinWeightValue();

	/** Maximum weight value
         * \return The maximum weight value in this neuron
         **/
	double getMaxWeightValue();

	/** Weight development
         * \return Returns the Euclidean distance of the weights from their starting position
         **/
	double getWeightDistanceFromInitialWeights();

	/** Gets the output of the neuron
         * \return The overall output of the neuron after the activation function
         **/
	inline double getOutput() { return output; };

	/** Gets the weighted sum of all inputs pre-activation function
         * \return Weighted sum (linear)
         **/
	inline double getSum() { return sum; };

	/** Gets one weight
         * \param _index The input index
         * \return The weight value at one input and one filter
         **/
	inline double getWeight( int _index) {
		assert((_index>=0)&&(_index<nInputs));
		return mask[_index] ? weights[_index] : 0;
	};

	/** Sets one weight
         * \param _index The input index
         * \param _weight The weight value
         **/
	inline void setWeight( int _index,  double _weight) {
		assert((_index>=0)&&(_index<nInputs));
		weights[_index]=_weight;
	};

	/** Sets the error in the neuron
         * If the derivative is activated then the derivative of the error is calculated.
         * \param _error Sets the error of the neuron.
         **/
	void setError(double _error);

	/** Gets the error as set by setError
         * \return The error value stored in the neuron
         **/
	inline double getError() { return error; };

	/** Sets one input 
         * \param _index Index of the input.
         * \param _value of the input.
         **/
	inline void setInput( int _index,  double _value) {
		assert((_index>=0)&&(_index<nInputs));
		inputs[_index] = _value;
	};

	/** Get the value at one input
         * \param _index Index of the input
         * \return Returns the input value
         **/
	inline double getInput( int _index) {
		assert((_index>=0)&&(_index<nInputs));
		return inputs[_index];
	};

	/** Gets the bias weight
         * \return Bias weight value
         **/
	inline double getBiasWeight() {return biasweight; };

	/** Sets the bias weight.
         * \param _biasweight Bias value.
         **/
	inline void setBiasWeight(double _biasweight) { biasweight=_biasweight; };

	/** Sets the bias input value.
         * \param _bias Bias value.
         **/
	inline void setBias( double _bias) { bias=_bias; };

	/** Sets the learning rate.
         * \param _learningrate The learning rate
         **/
	inline void setLearningRate( double _learningrate) { learningRate = _learningrate; };

	/** Sets the momentum.
         * Sets the inertia of the learning.
         * \param  _momentum The new momentum
         **/
	inline void setMomentum( double _momentum) { momentum = _momentum; };

	/** Sets the weight decay over time.
         * \param _decay The larger the faster the weight decay.
         **/
	inline void setDecay(double _decay) {
		decay = _decay;
	}

	/** Gets the weight decay over time.
         * \return The weight decay value. The larger the faster the weight decay.
         **/
	inline double getDecay() {
		return decay;
	}

	/** Get the number of inputs to the neuron
         * \return The numer of inputs
         **/
	inline int getNinputs() { return nInputs; };

	/** Tells the layer that it's been a 2D array originally to be a convolutional layer.
         * _width * _height == nInputs. Otherwise an exception is triggered.
         * The geometry entered here is then used in the mask operations so that every
         * neuron is able to process a subset of the input space, for example an image and
         * thus becomes a localised receptive field.
         * \param _width The width of the layer
         * \param _height of the layer
         **/
	void setGeometry( int _width,  int _height) {
		assert((_width*_height)==nInputs);
			width = _width;
			height = _height;
	}
	
	/** Boundary safe 2D manipulation of the convolution mask.
         * Sets the convolution mask using the geometry defined by setGeometry.
         * \param x Sets the mask value at coordinate x (0 .. width).
         * \param y Sets the mask value at coordinate y (0 .. height).
         * \param c Sets the mask: 0 = ignore underlying value, 1 = process underlying value.
         **/
	void setMask( int x, int y, unsigned char c);
	
	/** Boundary safe 1D manipulation of the convolution mask.
         * \param x Sets the mask value from coordinate x.
         * \param c Sets the mask: 0 = ignore underlying value, 1 = process underlying value.
         **/
	void setMask( const int x, const unsigned char c);
	
	/** Init the whole mask with a single value.
         * \param c Sets the mask for the whole array. 0 = ignore the entire input, 1 = process every input.
         **/
	void setMask( unsigned char c);
	
	/** Boundary safe return of the mask in (x,y) coordinates.
         * \param x Sets the mask value at coordinate x (0 .. width).
         * \param y Sets the mask value at coordinate y (0 .. height).
         * \return The mask at x,y: 0 = ignore underlying value, 1 = process underlying value.
         **/
	unsigned char getMask( int x, int y);

	/** Boundary safe return of the mask in flat form.
         * \param index Mask index.
         * \return The mask at the index: 0 = ignore underlying value, 1 = process underlying value.
         **/
	unsigned char getMask(int index) { return mask[index]; };

	/** Calculates the sum of the squared weight vector values.
	 * \return The squared weight vector values.
	 **/
	double getSumOfSquaredWeightVector();

	/** Calculates the Eucledian length of the weight vector
         * \return Eucledian length of the weight vector.
         **/
	double getEuclideanNormOfWeightVector() {
		return sqrt(getSumOfSquaredWeightVector());
	}

	/** Calculates the Manhattan length of the weight vector
         * /return Manhattan length of the weight vector.
         **/
	double getManhattanNormOfWeightVector();

	/** Calculates the Infinity norm of the vector.
         * /return Infinity norm of the vector.
         **/
	double getInfinityNormOfWeightVector();

	/** Calculates the average of the weight values.
         * \return average of the weight values.
         **/
	double getAverageOfWeightVector();

	/** Normalises the weights with a divisor.
         * \param norm Divisor which normalises the weights.
         **/
	void normaliseWeights(double norm);

	/** Save the initial weights.
         * This saves the initial weights for later comparisons. For internal use.
         **/
	void saveInitialWeights();

	/** Sets debug info populated from Layer.
         * \param _layerIndex The layer the neuron is in.
         * \param _neuronIndex The index of the neuron in the layer.
         **/
	void setDebugInfo(int _layerIndex, int _neuronIndex) {
		layerIndex = _layerIndex;
		neuronIndex = _neuronIndex;
	}

	/** Sets the simulation step for debugging and logging.
         * \param _step Current simulation step.
         **/
	inline void setStep(long int _step) {
		step = _step;
	}

private:
	int nInputs;
	unsigned char* mask = 0;
	double* weights = 0;
	double* initialWeights = 0;
	double* weightChange = 0;
	double decay = 0;
	double biasweight = 0;
	double biasweightChange = 0;
	double bias = 0;
	double* inputs = 0;
	double output = 0;
	double sum = 0;
	double error = 0;
	double learningRate = 0;
	double learningRateFactor = 1;
	double momentum = 0;
	int width = 0;
	int height = 0;
	int maxDet = 0;
	int layerIndex = 0;
	int neuronIndex = 0;
	long int step = 0;
	ActivationFunction activationFunction = TANH;
};

#endif
