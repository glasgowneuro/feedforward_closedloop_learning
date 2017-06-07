#include "neuron.h"
#include<stdio.h>

int main(int,char**) {
	Neuron* neuron = new Neuron(2);

	neuron->setWeight(0,10);
	neuron->setWeight(1,1);

	neuron->setInput(0,0);
	neuron->setInput(1,0);

	for (float f= -1;f<=1;f=f+0.1) {
		neuron->setInput(0,f);
		neuron->calcOutput();
		printf("%f %f %f\n",f,neuron->getSum(),neuron->getOutput());
	}
}
