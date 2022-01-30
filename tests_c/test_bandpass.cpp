#include "neuron.h"
#include<stdio.h>
#include<stdlib.h>
#include"bandpass.h"

int main(int n,char**args) {
        if (n<2) {
		fprintf(stderr,"Usage: %s <cutoff>\n",args[0]);
		exit(1);
	}
	
        double dampingCoeff = 0.51;
        Bandpass* bandpass = new Bandpass();
	bandpass->setParameters(atof(args[1]),dampingCoeff);
	char filename[]="impulse.dat";
	bandpass->impulse(filename);
	delete bandpass;
}


