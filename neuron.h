#ifndef __Neuron_H_
#define __Neuron_H_

// bypasses the sigmoid function
// #define DEBUG_OUTPUT

class Neuron {


public:

	Neuron(int _nInputs);
	~Neuron();
	void calcOutput();
	void doLearning();
	void initWeights(float _max);

	int nInputs;
	float* weights;
	float* inputs;
	float output;
	float sum;
	float bias;

	//

	float error;
	float learningRate;
};

#endif
