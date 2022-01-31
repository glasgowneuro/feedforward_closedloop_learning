#include "fcl.h"
#include<stdio.h>
#include <signal.h>
#include <stdio.h>
#include <signal.h>
#ifdef __linux__
#include <execinfo.h>
#endif

// inits the network with random weights with quite a few hidden units so that
// a nice response is generated
void test_forward() {
	printf("test_forward\n");
	int nNeuronsHidden[] = {10,10,1};

	FeedforwardClosedloopLearning fcl(2,nNeuronsHidden,3);
	fcl.seedRandom(1);
	FILE* f=fopen("test_fcl_cpp_forward.dat","wt");
	// no learning
	fcl.setLearningRate(0.0);
	// random init
	fcl.initWeights(1, 0, Neuron::MAX_OUTPUT_RANDOM);

	double input[2];
	double error[2];

	for(int n = 0; n < 100;n++) {

		input[0] = 0;
		input[1] = 0;
		if ((n>10)&&(n<20)) {
			input[0] = 0.1;
			input[1] = 0.1;
		}
		fprintf(f,"%f ",input[0]);

		fcl.doStep(input,error);
		for(int i=0; i<fcl.getNumLayers(); i++) {
			fprintf(f,"%e ",fcl.getLayer(i)->getNeuron(0)->getSum());
		}
		fprintf(f,"%e ",fcl.getOutputLayer()->getNeuron(0)->getOutput());
		fprintf(f,"\n");
	}

	fclose(f);
}



void test_learning_fcl() {
	printf("test_learning_fcl\n");
	int nNeur[] = {2,1};
	FeedforwardClosedloopLearning fcl(2,nNeur,2);
	fcl.seedRandom(1);
	fcl.setLearningRate(0.001);
	fcl.initWeights(1,0,Neuron::MAX_OUTPUT_RANDOM);
	fcl.setLearningRateDiscountFactor(1);
	fcl.setBias(0);
	
	FILE* f=fopen("test_learning_fcl.dat","wt");

	double input[2] = { 0,0 };
	double error[2] = { 0,0 };
	
	for(int n = 0; n < 10000;n++) {
		
		double stim = 0;
		double err = 0;
		int n2 = n% 1000;
		if ((n2>100)&&(n2<1000)) {
			stim = 1;
			if ((n2>500)&&(n2<600)) {
				err = 1;
			}
		}
		fprintf(f,"%f %f ",stim,err);

		input[0] = stim;
		error[0] = err;
		error[1] = err;

		fcl.doStep(input,error);

		for(int k=0; k<fcl.getNumLayers(); k++) {
			for(int i=0;i<fcl.getLayer(k)->getNneurons();i++) {
				for(int j=0;j<fcl.getLayer(k)->getNeuron(i)->getNinputs();j++) {
					fprintf(f, "%e ",
						fcl.getLayer(k)->getNeuron(i)->getWeight(j));
				}
			}
		}

		fprintf(f,
			"%e ",
			fcl.getOutputLayer()->getNeuron(0)->getOutput());
		
		fprintf(f,"\n");
	}

	fclose(f);
}


int main(int,char**) {
	test_forward();
	test_learning_fcl();
}
