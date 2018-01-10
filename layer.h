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
#include <stdlib.h>
#include <string.h>
#include <list>
#ifdef __linux__
#include <pthread.h>
#endif


#define NUM_THREADS 12


// abstract thread which contains the inner workings of the thread model
class LayerThread {

protected:

	Neuron** neurons;
	int nNeurons = 0;
	int maxNeurons = 0;
	pthread_t id = 0;

	static void *exec(void *thr) {
		reinterpret_cast<LayerThread *> (thr)->run();
		return NULL;
	}

public:
	LayerThread(int _maxNeurons) {
		maxNeurons = _maxNeurons;
		neurons = new Neuron*[maxNeurons];
	}

	~LayerThread() {
		delete [] neurons;
	}
	
	void addNeuron(Neuron* neuron) {
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
		int ret;
		if ((ret = pthread_create(&id, NULL, &LayerThread::exec, this)) != 0) { 
			fprintf(stderr,"%s\n",strerror(ret)); 
			throw "Error"; 
		}
	}

	void join() {
		if (nNeurons == 0) {
			return;
		}
		pthread_join(id,NULL);
	}

	// needs to be implemented
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

	// 0 = no Threads, 1 = Threads
	void setUseThreads(int _useThreads) {
		useThreads = _useThreads;
		if (!useThreads) {
			fprintf(stderr,"Thread execution if OFF\n");
		}
	};

	int saveWeightMatrix(char *filename);
	
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
	int useThreads = 1;
	CalcOutputThread** calcOutputThread = NULL;
	LearningThread** learningThread = NULL;
	MaxDetThread** maxDetThread = NULL;
};

#endif
