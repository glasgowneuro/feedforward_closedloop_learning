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
	void initWeights(double _max);
	inline double getOutput() { return output; };
	inline double getSum() { return sum; };
	inline double getWeight(int _index) { return weights[_index]; };
	inline void setWeight(int _index, double _weight) { weights[_index]=_weight; };
	inline void setError(double _error) { error=_error; };
	inline double getError() { return error; };
	inline void setInput(int _index, double _value) { inputs[_index] = _value; };
	inline double getInput(int _index) { return inputs[_index]; };
	inline void setBias(double _bias) { bias=_bias; };
	inline void setLearningRate(double _learningrate) { learningRate = _learningrate; };

private:
	int nInputs;
	double* weights;
	double* inputs;
	double output;
	double sum;
	double bias;
	double error;
	double learningRate;
};

#endif
