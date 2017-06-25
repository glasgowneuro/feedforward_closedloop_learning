#ifndef __Neuron_H_
#define __Neuron_H_

// bypasses the sigmoid function
// #define LINEAR_OUTPUT

// enables denbug output to sdt out
// #define DEBUG_NEURON

#include "bandpass.h"

class Neuron {

public:

	Neuron(int _nInputs, int _nFilters = 0, double _minT = 0, double _maxT = 0);
	~Neuron();
	void calcOutput();
	void doLearning();
	void initWeights(double _max);
	inline double getOutput() { return output; };
	inline double getSum() { return sum; };
	inline double getWeight(int _index, int _filter = 0) { return weights[_index][_filter]; };
	inline void setWeight(int _index, double _weight, int _filter = 0) { weights[_index][_filter]=_weight; };
	void setError(double _error);
	inline double getError() { return error; };
	inline void setInput(int _index, double _value) { inputs[_index] = _value; };
	inline double getInput(int _index) { return inputs[_index]; };
	inline void setBias(double _bias) { bias=_bias; };
	inline void setLearningRate(double _learningrate) { learningRate = _learningrate; };
	inline void setUseDerivative(int _useDerivative) { useDerivative = _useDerivative; };

private:
	int nInputs;
	int nFilters;
	double** weights;
	Bandpass ***bandpass;
	double* inputs;
	double output;
	double sum;
	double bias;
	double error;
	double learningRate;
	double minT,maxT;
	double dampingCoeff;
	int useDerivative;
	double oldError;
};

#endif
