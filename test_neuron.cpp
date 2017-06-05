#include "neuron.h"
#include<stdio.h>

int main(int,char**) {
	Neuron* neuron = new Neuron(2);

	neuron->weights[0] = 10;
	neuron->weights[1] = 1;

	neuron->inputs[0] = 0;
	neuron->inputs[1] = 0;

	for (float f= -1;f<=1;f=f+0.1) {
		neuron->inputs[0] = f;
		neuron->calcOutput();
		printf("%f %f %f\n",f,neuron->sum,neuron->output);
	}
}
