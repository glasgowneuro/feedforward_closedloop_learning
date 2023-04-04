#include "cldl_neuron.h"

#include <assert.h>
#include <iostream>
#include <ctgmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <numeric>
#include <vector>

using namespace std;

//*************************************************************************************
// constructor de-constructor
//*************************************************************************************

CLDLNeuron::CLDLNeuron(int _nInputs)
{
	nInputs=_nInputs;
	weights = new double[(unsigned)nInputs];
	initialWeights = new double[(unsigned)nInputs];
	inputs = new double[(unsigned)nInputs];
}

CLDLNeuron::~CLDLNeuron(){
	delete [] weights;
	delete [] initialWeights;
	delete [] inputs;
}

//*************************************************************************************
//initialisation:
//*************************************************************************************

void CLDLNeuron::initNeuron(int _neuronIndex, int _layerIndex, weightInitMethod _wim, biasInitMethod _bim, actMethod _am){
	myLayerIndex = _layerIndex;
	myNeuronIndex = _neuronIndex;
	actMet = _am;
	for (int i=0; i<nInputs; i++) {
		switch (_wim) {
		case W_ZEROS:
			weights[i] = 0;
			break;
		case W_ONES:
			weights[i] = 1;
			break;
		case W_ONES_NORM:
			weights[i] = 1.0 / (double)nInputs;
			break;
		case W_RANDOM:
			weights[i] = ((( (double)rand() / RAND_MAX)) - 0.5);
			break;
		case W_RANDOM_NORM:
			weights[i] = ((( (double)rand() / RAND_MAX)) - 0.5) / (double)nInputs;
			break;
		}
		initialWeights[i] = weights[i];
	}
	weightSum = 0;
	for (int i=0; i<nInputs; i++){
		weightSum += fabs(weights[i]);
		maxWeight = max(maxWeight, weights[i]);
		minWeight = min (minWeight, weights[i]);
	}

	switch (_bim){
        case B_NONE:
		bias=0;
		break;
        case B_RANDOM:
		bias= (( (double)rand() / (RAND_MAX))) - 0.5;
		break;
	}
}

void CLDLNeuron::setLearningRate(double _w_learningRate, double _b_learningRate){
	w_learningRate=_w_learningRate;
	b_learningRate = _b_learningRate;
}

//*************************************************************************************
//forward propagation of inputs:
//*************************************************************************************

void CLDLNeuron::setInput(int _index,  double _value) {
	/* the seInput function sets one input value at the given index,
	 * it has to be implemented in a loop inside the layer class to set
	 * all the inputs associated with all the neurons in that layer*/
	assert((_index>=0)&&(_index<nInputs));
	/*checking _index is a valid int, non-negative and within boundary*/
	inputs[_index] = _value;
}

void CLDLNeuron::propInputs(int _index,  double _value){
	/*works like setInput function expect it only applies
	 * to the neurons in the hidden and output layers
	 * and not the input layer*/
	assert((_index>=0)&&(_index<nInputs));
	inputs[_index] = _value;
}

void CLDLNeuron::calcOutput(){
	sum=0;
	for (int i=0; i<nInputs; i++){
		sum += inputs[i] * weights[i];
	}
	sum += bias;
	assert(std::isfinite(sum));
	output = doActivation(sum);
	assert(std::isfinite(output));
}


//*************************************************************************************
//back propagation of error
//*************************************************************************************

void CLDLNeuron::setError(double _backwardError){
	error = _backwardError;
	assert(std::isfinite(error));
	/*might take a different format to propError*/
}


double CLDLNeuron::getError(){
	return error;
}


void CLDLNeuron::updateWeights(){
	weightSum = 0;
	maxWeight = 0;
	minWeight = 0;
	for (int i=0; i<nInputs; i++){
		weights[i] += w_learningRate * inputs[i] * error;
		weightSum += fabs(weights[i]);
		maxWeight = max (maxWeight,weights[i]);
		minWeight = min (maxWeight,weights[i]);
	}
	bias += b_learningRate * error;
}

//*************************************************************************************
// getters:
//*************************************************************************************

double CLDLNeuron::getWeightChange(){
	weightsDifference = 0;
	weightChange = 0;
	for (int i=0; i<nInputs; i++){
		weightsDifference = weights[i] - initialWeights[i];
		weightChange += pow(weightsDifference,2);
	}
	return (weightChange);
}

double CLDLNeuron::getWeightDistance(){
	return sqrt(weightChange);
}

int CLDLNeuron::getnInputs(){
	return (nInputs);
}

double CLDLNeuron::getWeights(int _inputIndex){
	assert((_inputIndex>=0)&&(_inputIndex<nInputs));
	return (weights[_inputIndex]);
}

double CLDLNeuron::getInitWeights(int _inputIndex){
	assert((_inputIndex>=0)&&(_inputIndex<nInputs));
	return (initialWeights[_inputIndex]);
}

//*************************************************************************************
//saving and inspecting
//*************************************************************************************

void CLDLNeuron::saveWeights(){
	string name = "w";
	name += 'L';
	name += std::to_string(myLayerIndex + 1);
	name += 'N';
	name += std::to_string(myNeuronIndex + 1);
	name += ".csv";
	std::ofstream Icofile;
	Icofile.open(name, fstream::app);
	for (int i=0; i<nInputs; i++){
		Icofile << weights[i] << " " ;
	}
	Icofile << "\n";
	Icofile.close();
}

void CLDLNeuron::printNeuron(){
	cout<< "\t \t This neuron has " << nInputs << " inputs:";
	for (int i=0; i<nInputs; i++){
		cout<< " " << inputs[i];
	}
	cout<<endl;
	cout<< "\t \t The weights for those inputs are:";
	for (int i=0; i<nInputs; i++){
		cout<< " " << weights[i];
	}
	cout<<endl;
	cout<< "\t \t The bias of the neuron is: " << bias << endl;
	cout<< "\t \t The sum and output of this neuron are: " << sum << ", " << output << endl;
}
