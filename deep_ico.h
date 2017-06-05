#ifndef __Deep_ICO_H_
#define __Deep_ICO_H_


#include "layer.h"
#include "neuron.h"


class Deep_ICO {

public:

	Layer* hiddenLayer;
	Layer* outputLayer;

	int neuronsPerLayer;
	int nInputsPerNeuron;

	Deep_ICO(int _neuronsPerLayer, int _nInputsPerNeuron);
	~Deep_ICO();

	void doStep();

	void setLearningRate(float learningRate);
};

#endif
