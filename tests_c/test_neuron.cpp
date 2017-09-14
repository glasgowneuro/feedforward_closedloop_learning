#include "neuron.h"
#include<stdio.h>

int main(int,char**) {
	int nFilters = 10;
	
	Neuron* neuron = new Neuron(2,nFilters,10,200);

	for(int i=0;i<nFilters;i++) {
		neuron->setWeight(0,1,i);
	}

	neuron->setInput(0,0);
	neuron->setInput(1,0);

	float f = 0;
	for (int i=0;i<1000;i++) {
		f = 0;
		if ((i>100)&&(i<110)) {
			f = 1;
		}
		neuron->setInput(0,f);
		neuron->calcOutput();
		printf("%f %f %f\n",f,neuron->getSum(),neuron->getOutput());
	}
}
