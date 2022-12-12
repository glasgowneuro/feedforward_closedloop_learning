#ifndef __Layer_H_
#define __Layer_H_

/**
 * GNU GENERAL PUBLIC LICENSE
 * Version 3, 29 June 2007
 *
 * (C) 2017-2022, Bernd Porr <bernd@glasgowneuro.tech>
 * (C) 2017, Paul Miller <paul@glasgowneuro.tech>
 **/

#include "globals.h"
#include "neuron.h"
#include <stdlib.h>
#include <string.h>
#include <thread>

#define NUM_THREADS 12

#ifndef DOXYGEN_SHOULD_SKIP_THIS

// abstract thread which contains the inner workings of the thread model
class LayerThread {

protected:

	FCLNeuron** neurons;
	int nNeurons = 0;
	int maxNeurons = 0;

	std::thread thr;

public:

	LayerThread(int _maxNeurons) {
		maxNeurons = _maxNeurons;
		neurons = new FCLNeuron*[maxNeurons];
	}

	virtual ~LayerThread() {
		delete [] neurons;
	}
	
	void addNeuron(FCLNeuron* neuron) {
		if (nNeurons >= maxNeurons) {
			fprintf(stderr,"Not enough memory for threads.\n");
			exit(1);
		}
		neurons[nNeurons] = neuron;
		nNeurons++;
	}

	void start() {
		if (nNeurons == 0) {
			return;
		}
		thr = std::thread(&LayerThread::run,this);
	}

	void join() {
		if (nNeurons == 0) {
			return;
		}
		if (thr.joinable()) {
			thr.join();
		}
	}

	// is implemented by its children to do the specfic task the thread
	virtual void run() = 0;
};


class CalcOutputThread : public LayerThread {
	using LayerThread::LayerThread;
	void run() {
		for (int i=0;i<nNeurons;i++) {
			neurons[i]->calcOutput();
		}
	}
};


class LearningThread : public LayerThread {
	using LayerThread::LayerThread;
	void run() {
		for (int i=0;i<nNeurons;i++) {
			neurons[i]->doLearning();
		}
	}
};


class MaxDetThread : public LayerThread {
	using LayerThread::LayerThread;
	void run() {
		for (int i=0;i<nNeurons;i++) {
			neurons[i]->doMaxDet();
		}
	}
};


#endif /* DOXYGEN_SHOULD_SKIP_THIS */


/** Layer which contains the neurons of one layer.
 * It performs all computations possible in a
 * layer. In particular it calls all neurons in 
 * separate threads and triggers the compuations
 * there. These functions are all called from the
 * parent class.
 **/
class FCLLayer {
	
public:
	/** Constructor
         * \param _nNeurons Number of neurons in the layer.
         * \param _nInputs Number of inputs to the Layer.
         * \param _nFilters Number of lowpass filters at each input.
         * \param _minT Minimum time of the lowpass filter.
         * \param _maxT Maximum time of the lowpass filter.
         **/
	FCLLayer(int _nNeurons, int _nInputs);

	/** Destructor
         * Frees all memory
         **/
	~FCLLayer();

	/** Weight normalisation constants
         * Defines if weights are normalised layer-wide or
         * for every neuron separately.
         **/
	enum WeightNormalisation {
		WEIGHT_NORM_NONE = 0,
		WEIGHT_NORM_LAYER_EUCLEDIAN = 1,
		WEIGHT_NORM_NEURON_EUCLEDIAN = 2,
		WEIGHT_NORM_LAYER_MANHATTAN = 3,
		WEIGHT_NORM_NEURON_MANHATTAN = 4,
		WEIGHT_NORM_LAYER_INFINITY = 5,
		WEIGHT_NORM_NEURON_INFINITY = 6
	};

	/** Calculates the output values in all neurons
         **/
	void calcOutputs();

	/** Adjusts the weights
         **/
	void doLearning();

	/** Sets the global error for all neurons
         * \param _error Sets the error in the whole layer
         **/
	void setError( double _error);

	/** sets the error individually
         * \param i Index of the neuron
         * \param _error The error to be set
         **/
	void setError( int i,  double _error);

	/** Sets all errors from an input array
         * \param _errors is an array of errors
         **/
	void setErrors( double *_errors);

	/** Retrieves the error
         * \param i Index of the neuron
         **/
	double getError( int i);

	/** Sets the global bias for all neurons
         * \param _bias The bias for all neurons
         **/
	void setBias( double _bias);

        /** Set the input value of one input
         * \param inputIndex The index number of the input.
         * \param input The value of the input
         **/
	void setInput( int inputIndex,  double input);

	/** Sets all inputs from an input array
         * \param _inputs array of all inputs
         **/
	void setInputs( const double * _inputs);

	/** Sets the learning rate of all neurons
         * \param _learningRate The learning rate
         **/
	void setLearningRate( double _learningRate);

	/** Set the activation function
         * \param _activationFunction The activation function. See: Neuron::ActivationFunction
         **/
	void setActivationFunction(FCLNeuron::ActivationFunction _activationFunction);

	/** Set the momentum of all neurons in this layer
         * \param _momentum The momentum for all neurons in this layer.
         **/
	void setMomentum( double _momentum);

	/** Sets the weight decay scaled by the learning rate
         * \param _decay The decay rate of the weights
         **/
	void setDecay( double _decay);

	/** Inits the weights
         * \param _max Maximum value if using random init.
         * \param initBiasWeight if one also the bias weight is initialised.
         * \param weightInitMethod The methid employed to init the weights.
         **/
	void initWeights( double _max = 1,
			  int initBiasWeight = 1,
			  FCLNeuron::WeightInitMethod weightInitMethod = FCLNeuron::MAX_OUTPUT_RANDOM);
	
	/** Gets the outpuut of one neuron
         * \param index The index number of the neuron.
         * \return Retuns the double valye of the output.
         **/
	inline double getOutput( int index) {
		return neurons[index]->getOutput();
	}

	/** Gets a pointer to one neuron
         * \param index The index number of the neuron.
         * \return A pointer to a Layer class.
         **/
	inline FCLNeuron* getNeuron( int index) {
		assert(index < nNeurons);
		return neurons[index];
	}

	/** Gets the number of neurons
         * \return The number of neurons.
         **/
	inline int getNneurons() { return nNeurons;}

	/** Number of inputs
	 * \return The number of inputs
         **/
	inline int getNinputs() { return nInputs;}

	/** Defines a 2D geometry for the input layer of widthxheight
         * \param width The width of the convolutional window.
         * \param height The height of the convolution window.
         **/
	void setConvolution( int width,  int height);

	/** Maxium detection layer. Experimental.
         * This hasn't been implemented.
         **/
	void setMaxDetLayer(int _m) { maxDetLayer = _m; };

	/** Normalise the weights
         * \param _normaliseWeights Metod of normalisation.
         **/
	void setNormaliseWeights(WeightNormalisation _normaliseWeights);

	/** Sets the layer index within the whole network.
         * \param layerIndex The layer index in the whole network.
         **/
	void setDebugInfo(int layerIndex);

	/** Sets the simulation step in the layer for debug purposes.
         * \param step Step number.
         **/
	void setStep(long int step);

	/** Get weight distance from the start of the simulation
         * \return The distance from the initial (random) weight setup.
         **/
	double getWeightDistanceFromInitialWeights();

	/** Performs the weight normalisation
         **/
	void doNormaliseWeights();

	/** Sets if threads should be used
         * \param _useThreads 0 = no Threads, 1 = Threads
         **/
	void setUseThreads(int _useThreads) {
		useThreads = _useThreads;
		if (!useThreads) {
			fprintf(stderr,"Thread execution if OFF\n");
		}
	};

	/** Save weight matrix for documentation and debugging
         * \param filename The filename it should be saved to.
         **/
	int saveWeightMatrix(char *filename);

	
private:

	int nNeurons;
	int nInputs;
	FCLNeuron** neurons = 0;
	int maxDetLayer = 0;
	WeightNormalisation normaliseWeights = WEIGHT_NORM_NONE;
	int debugOutput = 0;
	// for debugging output
	int layerIndex = 0;
	long int step = 0;
	int useThreads = 1;
	CalcOutputThread** calcOutputThread = nullptr;
	LearningThread** learningThread = nullptr;
	MaxDetThread** maxDetThread = nullptr;
};

#endif
