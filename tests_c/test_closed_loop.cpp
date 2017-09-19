#include "deep_feedback_learning.h"
#include <Iir.h>
#include<stdio.h>
#include <signal.h>
#include <stdio.h>


#define IIRORDER 2

void test_closedloop() {
	// We have one input
	int nInputs = 1;
	// We have one output neuron
	int nOutputs = 1;
	// We have two hidden layers
	int nHiddenLayers = 1;
	// We set two neurons in the first hidden layer
	int nNeuronsInHiddenLayers[] = {2};
	// We set nFilters in the input
	int nFiltersInput = 10;
	// We set nFilters in the hidden unit
	int nFiltersHidden = 10;
	// Filterbank
	double minT = 100;
	double maxT = 1000;
	
	DeepFeedbackLearning* deep_fbl = new DeepFeedbackLearning(
			nInputs,
			nNeuronsInHiddenLayers,
			nHiddenLayers,
			nOutputs,
			nFiltersInput,
			nFiltersHidden,
			minT,
			maxT);

	deep_fbl->initWeights(0.000001,0,Neuron::MAX_OUTPUT_CONST);
	deep_fbl->setLearningRate(1E-13);
	deep_fbl->setNormaliseLearningRateTo(1);
	deep_fbl->setAlgorithm(DeepFeedbackLearning::ico);
	deep_fbl->setBias(0);
	deep_fbl->setUseDerivative(1);
	//deep_fbl->getLayer(1)->setNormaliseWeights(1);
	
	Iir::Bessel::LowPass<IIRORDER> p0;
	p0.setup (IIRORDER,1,0.005);
	
	Iir::Bessel::LowPass<IIRORDER> h0;
	h0.setup (IIRORDER,1,0.1);
	
	FILE* f=fopen("test_closed_loop.dat","wt");

	double input[nInputs];
	double error[nNeuronsInHiddenLayers[0]];

	float v = 0;
	float v0 = 0;
	float x0 = 0;
	float err = 0;
	float dist = 0;
	float pred = 0;

	float fb_gain = 5;
	
	for(int step = 0; step < 20000; step++) {

		int n = step % 1000;
		
		pred = 0;
		dist = 0;
		if ((n>100)&&(n<1000)) {
			pred = 0;
			if (n<200) {
				pred = 1;
			}
			if ((n>500)&&(n<800)) {
				dist = 1;
			}
			if ((n>700)&&(n<1000)) {
				dist = 0;
			}
		}

		for(int i=0;i<nInputs;i++) {
			input[i] = pred;
		}
		
		for(int i=0;i<nNeuronsInHiddenLayers[0];i++) {
			error[i] = err;
		}
		
		deep_fbl->doStep(input,error);

		// error signal
		float setpoint = 0;
		err = (setpoint - x0) * fb_gain;

		// feedback filter plus the learned one
		v = h0.filter(err) + 1000* deep_fbl->getOutputLayer()->getNeuron(0)->getOutput();

		// the output of the controller plus disturbance
		v0 = dist + v;

		// that goes through the environment p0 and generates the input to the
		// controller
		x0 = p0.filter(v0);
		
		fprintf(f,"%d %f %f %f %f ",step,pred,dist,err,v);

		fprintf(f,
			"%f ",
			deep_fbl->getOutputLayer()->getNeuron(0)->getOutput());

		for(int i=0;i<nNeuronsInHiddenLayers[0];i++) {
			for(int j=0;j<nInputs;j++) {
				fprintf(f,
					"%e ",
					deep_fbl->getLayer(0)->getNeuron(i)->getWeight(j));
			}
		}
		fprintf(f,
			"%e ",
			deep_fbl->getOutputLayer()->getNeuron(0)->getWeight(0));
		fprintf(f,
			"%e ",
			deep_fbl->getOutputLayer()->getNeuron(0)->getWeight(1));
		fprintf(f,"\n");
		
	}

	fclose(f);
}


int main(int n,char** args) {
	test_closedloop();
}
