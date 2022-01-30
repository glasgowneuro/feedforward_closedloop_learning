#include "fcl.h"
#include<stdio.h>
#include <signal.h>
#include <stdio.h>
#include <signal.h>
#ifdef __linux__
#include <execinfo.h>
#endif

void runTest(int _useThreads) {
	int nNeurons = 10;
	int nInputs = 2;
	srand(1);
	Layer layer(nNeurons,nInputs);
	layer.setUseThreads(_useThreads);
	layer.initWeights(1, 0, Neuron::MAX_OUTPUT_RANDOM);
	layer.setError(0.1);
	layer.setInput(0,0.1);
	layer.setInput(1,0.1);
	layer.setLearningRate(1);

	for(int i=0;i<5;i++) {
			layer.calcOutputs();
			layer.doLearning();
			for(int j=0;j<nNeurons;j++) {
				fprintf(stderr,"%f ",layer.getNeuron(j)->getOutput());
			}
			fprintf(stderr,"\n");
	}
}


int main(int,char**) {
	fprintf(stderr,"Running Layer test with Threads turned off:\n");
	runTest(0);
	fprintf(stderr,"Running Layer test with Threads turned on:\n");
	runTest(1);
	return 0;
}
