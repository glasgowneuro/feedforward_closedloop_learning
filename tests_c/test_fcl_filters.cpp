#include "fcl_util.h"
#include <stdio.h>
#include <signal.h>
#include <stdio.h>
#include <signal.h>
#ifdef __linux__
#include <execinfo.h>
#endif


void test_filters() {
	printf("test_filters\n");
	int nNeur[] = {2,1};
	int nFiltersInput = 10;
	double minT = 10;
	double maxT = 200;
	FeedforwardClosedloopLearningWithFilterbank fcl(2,nNeur,2,nFiltersInput,minT,maxT);
	fcl.seedRandom(1);
	fcl.setLearningRate(0.001);
	fcl.initWeights(1,0,Neuron::MAX_OUTPUT_RANDOM);
	fcl.setLearningRateDiscountFactor(1);
	fcl.setBias(0);
	
	FILE* f=fopen("test_filters.dat","wt");

	double input[2] = { 0,0 };
	double error[2] = { 0,0 };

	for(int n = 0; n < 10000;n++) {		
		double stim = 0;
		const int n2 = n% 1000;
		if ((n2>100)&&(n2<200)) {
			stim = 1;
		}
		fprintf(f,"%f ",stim);

		input[0] = stim;
		input[1] = stim;

		fcl.doStep(input,error);

		for(int i=0;i<nFiltersInput;i++) {
			fprintf(f,"%e ",fcl.getFilterOutput(0,i));
		}

		fprintf(f,"\n");
	}
	fclose(f);
}


void test_learning_fcl_filters() {
	printf("test_learning_fcl_filters\n");
	int nNeur[] = {2,2,1};
	int nFiltersInput = 10;
	double minT = 100;
	double maxT = 1000;
	FeedforwardClosedloopLearningWithFilterbank fcl(2,nNeur,3,nFiltersInput,minT,maxT);
	fcl.seedRandom(1);
	fcl.setLearningRate(0.001);
	fcl.initWeights(1,0,Neuron::MAX_OUTPUT_RANDOM);
	fcl.setLearningRateDiscountFactor(1);
	fcl.setBias(0);
	
	FILE* f=fopen("test_learning_fcl_filters.dat","wt");
	FILE* f2=fopen("test_learning_fcl_filters2.dat","wt");

	double input[2] = { 0,0 };
	double error[2] = { 0,0 };

	for(int n = 0; n < 10000;n++) {
		
		double stim = 0;
		double err = 0;
		int n2 = n% 1000;
		if ((n2>400)&&(n2<7000)) {
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

		for(int i=0;i<nFiltersInput;i++) {
			fprintf(f2,"%e ",fcl.getFilterOutput(0,i));
		}
		fprintf(f2,"\n");

		
	}
	fclose(f);
	fclose(f2);
}


int main(int,char**) {
	test_filters();
	test_learning_fcl_filters();
}
