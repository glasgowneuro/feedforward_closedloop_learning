#ifndef __Deep_ICO_H_
#define __Deep_ICO_H_


#include "layer.h"
#include "neuron.h"

#define DO_DERIV

class Deep_ICO {

public:

        int ni;
        int nh;
        int no;

 	Layer* hiddenLayer;
	Layer* outputLayer;

	Deep_ICO(int num_input, int num_hidden, int num_output);
	~Deep_ICO();

	void doStep(float* input, float* error);

	void doStep(float* input, int n1, float* error, int n2) {
		doStep(input,error);
	}

	float getOutput(int index) {
		return outputLayer->getOutput(index);
	}

	void setLearningRate(float learningRate);

	void initWeights(float max);

#ifdef DO_DERIV
	float dsigm(float y) { return (1.0 - y*y); };
#else
	float dsigm(float y) { return y; };
#endif
};

#endif
