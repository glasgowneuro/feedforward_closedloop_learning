#ifndef __Deep_FEEDBACK_LEARNING_H_
#define __Deep_FEEDBACK_LEARNING_H_


#include "layer.h"
#include "neuron.h"
#include <stdio.h>
#include <stdlib.h>



// do the proper derivative of the activation function
#define DO_DERIV_ACTIVATION

//#define DEBUG

class DeepFeedbackLearning {

public:
	// deep ico without any filters
	DeepFeedbackLearning(int num_input, int num_hidden, int num_output);

	// deep ico with filters for both the input and hidden layer
	// filter number >0 means: filterbank
	// filter number = 0 means layer without filters
	// filter parameters: are in time steps. For ex, minT = 10 means
	// a response of 10 time steps for the first filter and that goes
	// up to maxT time steps, for example maxT = 100 or so.
	DeepFeedbackLearning(int num_input, int num_hidden, int num_output,
		 int num_filtersInput, int num_filtersHidden,
		 double _minT, double _maxT);
	
	~DeepFeedbackLearning();

	enum Algorithm { backprop = 0, ico = 1 };

	void doStep(double* input, double* error);

	inline void doStep(double* input, int n1, double* error, int n2) {
#ifdef DEBUG
		fprintf(stderr,"n1=%d,n2=%d\n",n1,n2);
#endif
		if (n1 != ni) {
			fprintf(stderr,"Input array dim mismatch: got: %d, want: %d\n",n1,ni);
			return;
		}
		if (n2 != no) {
			fprintf(stderr,"Error array dim mismatch: got: %d, want: %d\n",n2,no);
			return;
		}
		doStep(input,error);
	}

	double getOutput(int index) {
		return outputLayer->getOutput(index);
	}

	void setLearningRate(double learningRate);

	void setAlgorithm(Algorithm _algorithm) { algorithm = _algorithm; }

	Algorithm getAlgorithm() { return algorithm; }

	void initWeights(double max);

	void seedRandom(int s) { srandom(s); };

	Layer* getHiddenLayer() {return hiddenLayer;};
	Layer* getOutputLayer() {return outputLayer;};

	void setUseDerivative(int useIt);

private:

        int ni;
        int nh;
        int no;
	int nfInput;
	int nfHidden;
	double minT,maxT;

 	Layer* hiddenLayer;
	Layer* outputLayer;

	Algorithm algorithm;

#ifdef DO_DERIV_ACTIVATION
	double dsigm(double y) { return (1.0 - y*y); };
#else
	double dsigm(double y) { return y; };
#endif
};

#endif
