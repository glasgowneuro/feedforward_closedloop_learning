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
	int nFiltersInput = 10;
	int nFiltersHidden = 10;
	int nNeuronsHidden[] = {10,10};

	FeedbackClosedloopLearning* fcl = new FeedbackClosedloopLearning(2,nNeuronsHidden,2,1,nFiltersInput,nFiltersHidden,100,200);
	fcl->seedRandom(1);
	FILE* f=fopen("test_fcl_cpp_forward.dat","wt");
	// no learning
	fcl->setLearningRate(0.0);
	// random init
	fcl->initWeights(1, 0, Neuron::MAX_OUTPUT_RANDOM);

	double input[2];
	double error[2];

	for(int n = 0; n < 1000;n++) {

		input[0] = 0;
		input[1] = 0;
		if ((n>10)&&(n<20)) {
			input[0] = 0.1;
			input[1] = 0.1;
		}
		fprintf(f,"%f ",input[0]);

		fcl->doStep(input,error);
		for(int i=0; i<fcl->getNumHidLayers(); i++) {
			fprintf(f,"%e ",fcl->getLayer(i)->getNeuron(0)->getSum());
		}
		fprintf(f,"%e ",fcl->getOutputLayer()->getNeuron(0)->getOutput());
		
		fprintf(f,"\n");
		
	}

	fclose(f);
}



void test_learning_fcl() {
	int nHidden[] = {2};
	FeedbackClosedloopLearning* fcl = new FeedbackClosedloopLearning(2,nHidden,1,1);
	fcl->seedRandom(1);
	fcl->setLearningRate(0.001);
	fcl->initWeights(1,0,Neuron::MAX_OUTPUT_RANDOM);
	fcl->setLearningRateDiscountFactor(1);
	fcl->setBias(0);
	fcl->setUseDerivative(0);

	
	FILE* f=fopen("test_fcl_cpp_learning.dat","wt");

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

		fcl->doStep(input,error);

		for(int i=0;i<2;i++) {
			for(int j=0;j<2;j++) {
				for(int k=0; k<fcl->getNumHidLayers(); k++) {
					fprintf(f, "%e ",
							fcl->getLayer(k)->getNeuron(i)->getAvgWeight(j));
				}
			}
		}
		for(int i=0;i<1;i++) {
			for(int j=0;j<2;j++) {
				fprintf(f,
					"%e ",
					fcl->getOutputLayer()->getNeuron(i)->getAvgWeight(j));
			}
		}
		for(int i=0;i<1;i++) {
			fprintf(f,
				"%e ",
				fcl->getOutputLayer()->getNeuron(i)->getOutput());
		}
		fprintf(f,"\n");
		
	}

	fclose(f);
}


void test_learning_fcl_filters() {
	int nHidden[] = {2};
	FeedbackClosedloopLearning* fcl = new FeedbackClosedloopLearning(2,nHidden,1,1);
	fcl->seedRandom(1);
	fcl->setLearningRate(0.001);
	fcl->initWeights(1,0,Neuron::MAX_OUTPUT_RANDOM);
	fcl->setLearningRateDiscountFactor(1);
	fcl->setBias(0);
	fcl->setUseDerivative(0);

	
	FILE* f=fopen("test_fcl_cpp_learning.dat","wt");

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

		fcl->doStep(input,error);

		for(int i=0;i<2;i++) {
			for(int j=0;j<2;j++) {
				for(int k=0; k<fcl->getNumHidLayers(); k++) {
					fprintf(f, "%e ",
							fcl->getLayer(k)->getNeuron(i)->getAvgWeight(j));
				}
			}
		}
		for(int i=0;i<1;i++) {
			for(int j=0;j<2;j++) {
				fprintf(f,
					"%e ",
					fcl->getOutputLayer()->getNeuron(i)->getAvgWeight(j));
			}
		}
		for(int i=0;i<1;i++) {
			fprintf(f,
				"%e ",
				fcl->getOutputLayer()->getNeuron(i)->getOutput());
		}
		fprintf(f,"\n");
		
	}

	fclose(f);
}


int main(int n,char** args) {
	if (n<2) {
		fprintf(stderr,"%s <number>:\n",args[0]);
		fprintf(stderr,"0=network test / no learning\n");
		fprintf(stderr,"1=fcl learning w/o filters\n");
		fprintf(stderr,"2=fcl learning with filters\n");
		exit(0);
	}
	switch (atoi(args[1])) {
	case 0:
		test_forward();
		break;
	case 1:
		test_learning_fcl();
		break;
	case 2:
		test_learning_fcl_filters();
		break;
	}
}
