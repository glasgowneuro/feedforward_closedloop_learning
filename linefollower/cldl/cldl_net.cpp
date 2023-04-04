#include "cldl_net.h"
#include "cldl_layer.h"
#include "cldl_neuron.h"

#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <ctgmath>
#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <fstream>
#include <iostream>
#include <math.h>
#include <fstream>
#include <iostream>
#include <string>
#include <numeric>
#include <vector>

using namespace std;

//*************************************************************************************
//initialisation:
//*************************************************************************************

CLDLNet::CLDLNet(const int _nLayers, const int * const _nNeurons, const int _nInputs){
	nLayers = _nLayers; //no. of layers including inputs and outputs layers
	layers= new CLDLLayer*[(unsigned)nLayers];
	const int* nNeuronsp = _nNeurons; //number of neurons in each layer
	nInputs=_nInputs; // the no. of inputs to the network (i.e. the first layer)
	//cout << "nInputs: " << nInputs << endl;
	int nInput = 0; //temporary variable to use within the scope of for loop
	for (int i=0; i<nLayers; i++){
		int numNeurons= *nNeuronsp; //no. neurons in this layer
		if (i==0){nInput=nInputs;}
		/* no. inputs to the first layer is equal to no. inputs to the network */
		layers[i]= new CLDLLayer(numNeurons, nInput);
		nNeurons += numNeurons;
		nWeights += (numNeurons * nInput);
		nInput=numNeurons;
		/*no. inputs to the next layer is equal to the number of neurons
		 * in the current layer. */
		nNeuronsp++; //point to the no. of neurons in the next layer
	}
	nOutputs=layers[nLayers-1]->getnNeurons();
	errorGradient= new double[(unsigned)nLayers];
	//cout << "net" << endl;
}

CLDLNet::~CLDLNet(){
	for (int i=0; i<nLayers; i++){
		delete layers[i];
	}
	delete[] layers;
	delete[] errorGradient;
}

void CLDLNet::initNetwork(CLDLNeuron::weightInitMethod _wim, CLDLNeuron::biasInitMethod _bim, CLDLNeuron::actMethod _am){
	for (int i=0; i<nLayers; i++){
		layers[i]->initLayer(i, _wim, _bim, _am);
	}
}

void CLDLNet::setLearningRate(double _w_learningRate, double _b_learningRate){
	for (int i=0; i<nLayers; i++){
		layers[i]->setlearningRate(_w_learningRate, _b_learningRate);
	}
}

void CLDLNet::doStep(const double* input, const double* errors) {
	setInputs(input);
	setErrors(errors);
	propInputs();
	propErrorBackward();
	updateWeights();
}

//*************************************************************************************
//forward propagation of inputs:
//*************************************************************************************

void CLDLNet::setInputs(const double* _inputs) {
	inputs=_inputs;
	layers[0]->setInputs(inputs);
}

void CLDLNet::propInputs(){
	for (int i=0; i<nLayers-1; i++){
		layers[i]->calcOutputs();
		for (int j=0; j<layers[i]->getnNeurons(); j++){
			double inputOuput = layers[i]->getOutput(j);
			layers[i+1]->propInputs(j, inputOuput);
		}
	}
	layers[nLayers-1]->calcOutputs();
}

void CLDLNet::setErrors(const double* leadError){
	layers[nLayers-1]->setErrors(leadError);
}

void CLDLNet::propErrorBackward(){
	double tempError = 0;
	double tempWeight = 0;
	for (int i = nLayers-1; i > 0 ; i--){
		for (int k = 0; k < layers[i-1]->getnNeurons(); k++){
			double sum = 0.0;
			for (int j = 0; j < layers[i]->getnNeurons(); j++){
				tempError = layers[i]->getNeuron(j)->getError();
				tempWeight = layers[i]->getWeights(j,k);
				sum += (tempError * tempWeight);
			}
			assert(std::isfinite(sum));
			layers[i-1]->getNeuron(k)->setBackpropError(sum);
		}
	}
}

//*************************************************************************************
//exploding/vanishing gradient:
//*************************************************************************************

double CLDLNet::getGradient(CLDLLayer::whichGradient _whichGradient) {
	for (int i=0; i<nLayers; i++) {
		errorGradient[i] = layers[i]->getGradient(_whichGradient);
	}
	double gradientRatio = errorGradient[nLayers -1] / errorGradient[0] ; ///errorGradient[0];
	assert(std::isfinite(gradientRatio));
	return gradientRatio;
}

//*************************************************************************************
//learning:
//*************************************************************************************

void CLDLNet::updateWeights(){
	for (int i=nLayers-1; i>=0; i--){
		layers[i]->updateWeights();
	}
}

//*************************************************************************************
// getters:
//*************************************************************************************

double CLDLNet::getOutput(int _neuronIndex){
	return (layers[nLayers-1]->getOutput(_neuronIndex));
}

double CLDLNet::getSumOutput(int _neuronIndex){
	return (layers[nLayers-1]->getSumOutput(_neuronIndex));
}

int CLDLNet::getnLayers(){
	return (nLayers);
}

int CLDLNet::getnInputs(){
	return (nInputs);
}

CLDLLayer* CLDLNet::getLayer(int _layerIndex){
	assert(_layerIndex<nLayers);
	return (layers[_layerIndex]);
}

double CLDLNet::getWeightDistance(){
	double weightChange = 0 ;
	double weightDistance =0;
	for (int i=0; i<nLayers; i++){
		weightChange += layers[i]->getWeightChange();
	}
	weightDistance=sqrt(weightChange);
	// cout<< "Net: WeightDistance is: " << weightDistance << endl;
	return weightDistance;
}

double CLDLNet::getLayerWeightDistance(int _layerIndex){
	return layers[_layerIndex]->getWeightDistance();
}

double CLDLNet::getWeights(int _layerIndex, int _neuronIndex, int _weightIndex){
	double weight=layers[_layerIndex]->getWeights(_neuronIndex, _weightIndex);
	return (weight);
}

int CLDLNet::getnNeurons(){
	return (nNeurons);
}

//*************************************************************************************
//saving and inspecting
//*************************************************************************************

void CLDLNet::saveWeights(){
	for (int i=0; i<nLayers; i++){
		layers[i]->saveWeights();
	}
}


void CLDLNet::snapWeights(string name){
	for (int i=0; i<nLayers; i++){
		layers[i]->snapWeights(name);
	}
}

void CLDLNet::snapWeightsMatrixFormat(string prefix){
	layers[0]->snapWeightsMatrixFormat(prefix);
}

void CLDLNet::printNetwork(){
	cout<< "This network has " << nLayers << " layers" <<endl;
	for (int i=0; i<nLayers; i++){
		cout<< "Layer number " << i << ":" <<endl;
		layers[i]->printLayer();
	}
	cout<< "The output(s) of the network is(are):";
	for (int i=0; i<nOutputs; i++){
		cout<< " " << this->getOutput(i);
	}
	cout<<endl;
}
