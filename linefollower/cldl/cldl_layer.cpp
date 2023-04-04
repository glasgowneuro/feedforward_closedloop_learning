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
#include <fstream>

//*************************************************************************************
// constructor de-constructor
//*************************************************************************************

CLDLLayer::CLDLLayer(int _nNeurons, int _nInputs){
    nNeurons = _nNeurons; // number of neurons in this layer
    nInputs = _nInputs; // number of inputs to each neuron
    neurons = new CLDLNeuron*[(unsigned)nNeurons];
    /* dynamic allocation of memory to n number of
     * neuron-pointers and returning a pointer, "neurons",
     * to the first element */
    for (int i=0;i<nNeurons;i++){
        neurons[i]=new CLDLNeuron(nInputs);
    }
    /* each element of "neurons" pointer is itself a pointer
     * to a neuron object with specific no. of inputs*/
     //cout << "layer" << endl;
}

CLDLLayer::~CLDLLayer(){
    for(int i=0;i<nNeurons;i++) {
        delete neurons[i];
    }
    delete[] neurons;
    //delete[] inputs;
    /* it is important to delete any dynamic
     * memory allocation created by "new" */
}

//*************************************************************************************
//initialisation:
//*************************************************************************************

void CLDLLayer::initLayer(int _layerIndex, CLDLNeuron::weightInitMethod _wim, CLDLNeuron::biasInitMethod _bim, CLDLNeuron::actMethod _am){
    myLayerIndex = _layerIndex;
    for (int i=0; i<nNeurons; i++){
        neurons[i]->initNeuron(i, myLayerIndex, _wim, _bim, _am);
    }
}

void CLDLLayer::setlearningRate(double _w_learningRate, double _b_learningRate){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->setLearningRate(_w_learningRate,_b_learningRate);
    }
}

//*************************************************************************************
//forward propagation of inputs:
//*************************************************************************************

void CLDLLayer::setInputs(const double* const _inputs) {
	/*this is only for the first layer*/
	const double* inputs = _inputs;
	for (int j=0; j< nInputs; j++){
		CLDLNeuron** neuronsp = neurons;//point to the 1st neuron
		/* sets a temporarily pointer to neuron-pointers
		 * within the scope of this function. this is inside
		 * the loop, so that it is set to the first neuron
		 * everytime a new value is distributed to neurons */
		const double input= (*inputs); //take this input value
		for (int i=0; i<nNeurons; i++){
		  (*neuronsp)->setInput(j,input);
			//set this input value for this neuron
			neuronsp++; //point to the next neuron
		}
		inputs++; //point to the next input value
	}
}

void CLDLLayer::propInputs(int _index, double _value){
	for (int i=0; i<nNeurons; i++){
		neurons[i]->propInputs(_index, _value);
	}
}

void CLDLLayer::calcOutputs(){
	for (int i=0; i<nNeurons; i++){
		neurons[i]->calcOutput();
	}
}

//*************************************************************************************
//back propagation of error:
//*************************************************************************************

void CLDLLayer::setErrors(const double* backwardError){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->setError(backwardError[i]);
    }
}

//*************************************************************************************
//exploding/vanishing gradient:
//*************************************************************************************

double CLDLLayer::getGradient(whichGradient _whichGradient) {
    double averageError = 0;
    double maxError = -100;
    double minError = 100;
    switch(_whichGradient){
        case exploding:
            for (int i=0; i<nNeurons; i++){
                maxError = max(maxError, neurons[i]->getError());
            }
            return maxError;
            break;
        case average:
            for (int i=0; i<nNeurons; i++){
                averageError += neurons[i]->getError();
            }
            return averageError/nNeurons;
            break;
        case vanishing:
            for (int i=0; i<nNeurons; i++){
                minError = min(minError, neurons[i]->getError());
            }
            return minError;
            break;
    }
    return 0;
}

//*************************************************************************************
//learning:
//*************************************************************************************

void CLDLLayer::updateWeights(){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->updateWeights();
    }
}

//*************************************************************************************
//getters:
//*************************************************************************************

CLDLNeuron* CLDLLayer::getNeuron(int _neuronIndex){
    assert(_neuronIndex < nNeurons);
    return (neurons[_neuronIndex]);
}

double CLDLLayer::getSumOutput(int _neuronIndex){
    return (neurons[_neuronIndex]->getSumOutput());
}

double CLDLLayer::getWeights(int _neuronIndex, int _weightIndex){
    return (neurons[_neuronIndex]->getWeights(_weightIndex));
}

double CLDLLayer::getInitWeight(int _neuronIndex, int _weightIndex){
    return (neurons[_neuronIndex]->getInitWeights(_weightIndex));
}

double CLDLLayer::getWeightChange(){
    double weightChange=0;
    for (int i=0; i<nNeurons; i++){
        weightChange += neurons[i]->getWeightChange();
    }
    //cout<< "Layer: WeightChange is: " << weightChange << endl;
    return weightChange;
}

double CLDLLayer::getWeightDistance(){
    return sqrt(getWeightChange());
}

double CLDLLayer::getOutput(int _neuronIndex){
    return (neurons[_neuronIndex]->getOutput());
}

int CLDLLayer::getnNeurons(){
    return (nNeurons);
}

//*************************************************************************************
//saving and inspecting
//*************************************************************************************

void CLDLLayer::saveWeights(){
    for (int i=0; i<nNeurons; i++){
        neurons[i]->saveWeights();
    }
}

void CLDLLayer::snapWeights(string name){
    std::fstream wfile;
    wfile.open(name, fstream::out);
    if (!wfile || !wfile) {
        cout << "Unable to open grayScale files";
        exit(1); // terminate with error
    }
    for (int i=0; i<nNeurons; i++){
        for (int j=0; j<nInputs; j++){
            wfile << neurons[i]->getWeights(j) << " ";
        }
        wfile << "\n";
    }
    wfile.close();
}

void CLDLLayer::snapWeightsMatrixFormat(string name){
    std::ofstream wfile;
    wfile.open(name);
    wfile << "[" << nNeurons << "," << nInputs << "]";
    wfile << "(";
    for (int i=0; i<nNeurons; i++){
        if (i == 0){
            wfile << "(";
        }else{
            wfile << ",(";
        }
        for (int j=0; j<nInputs; j++){
            if (j == 0){
                wfile << neurons[i]->getWeights(j);
            }else{
                wfile << "," << neurons[i]->getWeights(j);
            }
        }
        wfile << ")";
        //wfile << "\n";
    }
    wfile << ")";
    wfile.close();
}

void CLDLLayer::printLayer(){
    cout<< "\t This layer has " << nNeurons << " Neurons" <<endl;
    cout<< "\t There are " << nInputs << " inputs to this layer" <<endl;
    for (int i=0; i<nNeurons; i++){
        cout<< "\t Neuron number " << i << ":" <<endl;
        neurons[i]->printNeuron();
    }

}
