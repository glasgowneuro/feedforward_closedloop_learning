#ifndef __Neuron_H_
#define __Neuron_H_

// bypasses the sigmoid function
// #define LINEAR_OUTPUT

class Neuron {


public:

	Neuron(int _nInputs);
	~Neuron();
	void calcOutput();
	void doLearning();
	void initWeights(float _max);
	inline float getOutput() { return output; };
	inline float getSum() { return sum; };
	inline float getWeight(int _index) { return weights[_index]; };
	inline void setWeight(int _index, float _weight) { weights[_index]=_weight; };
	inline void setError(float _error) { error=_error; };
	inline float getError() { return error; };
	inline void setInput(int _index, float _value) { inputs[_index] = _value; };
	inline float getInput(int _index) { return inputs[_index]; };
	inline void setBias(float _bias) { bias=_bias; };
	inline void setLearningRate(float _learningrate) { learningRate = _learningrate; };

private:
	int nInputs;
	float* weights;
	float* inputs;
	float output;
	float sum;
	float bias;
	float error;
	float learningRate;
};

#endif
