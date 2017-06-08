#include "deep_ico.h"
#include <Iir.h>
#include<stdio.h>

void test_forward() {
	Deep_ICO* deep_ico = new Deep_ICO(2,2,2);
	FILE* f=fopen("test_deep_ico_cpp_forward.dat","wt");

	deep_ico->setLearningRate(0.0);
	
	double input[2];
	double error[2];

	deep_ico->getOutputLayer()->getNeuron(0)->setWeight(0,1);
	deep_ico->getHiddenLayer()->getNeuron(0)->setWeight(0,1);
	deep_ico->getHiddenLayer()->getNeuron(0)->setWeight(1,1);

	for(int n = 0; n < 100;n++) {

		input[0] = 0;
		input[1] = 1;
		if ((n>10)&&(n<20)) {
			input[0] = 1;
			input[1] = 1;
		}
		fprintf(f,"%f ",input[0]);

		deep_ico->doStep(input,error);

		fprintf(f,"%f ",deep_ico->getHiddenLayer()->getNeuron(0)->getSum());
		fprintf(f,"%f ",deep_ico->getHiddenLayer()->getNeuron(1)->getSum());
		fprintf(f,"%f ",deep_ico->getOutputLayer()->getNeuron(0)->getOutput());
		fprintf(f,"%f ",deep_ico->getOutputLayer()->getNeuron(1)->getOutput());
		
		fprintf(f,"\n");
		
	}

	fclose(f);
}



void test_learning() {
	Deep_ICO* deep_ico = new Deep_ICO(2,2,1);
	deep_ico->initWeights(0.01);
	
	FILE* f=fopen("test_deep_ico_cpp_learning.dat","wt");

	double input[2];
	double error[2];	
	
	for(int n = 0; n < 1000;n++) {
		
		double stim = 0;
		double err = 0;
		if ((n>10)&&(n<1000)) {
			stim = 1;
			if ((n>500)&&(n<600)) {
				err = 1;
			}
			if ((n>700)&&(n<1000)) {
				err = -1;
			}
		}
		fprintf(f,"%f %f ",stim,err);

		input[0] = stim;
		error[0] = err;

		deep_ico->doStep(input,error);

		for(int i=0;i<2;i++) {
			for(int j=0;j<2;j++) {
				fprintf(f,
					"%f ",
					deep_ico->getHiddenLayer()->getNeuron(i)->getWeight(j));
			}
		}
		for(int i=0;i<1;i++) {
			for(int j=0;j<2;j++) {
				fprintf(f,
					"%f ",
					deep_ico->getOutputLayer()->getNeuron(i)->getWeight(j));
			}
		}
		for(int i=0;i<1;i++) {
			fprintf(f,
				"%f ",
				deep_ico->getOutputLayer()->getNeuron(i)->getOutput());
		}
		fprintf(f,"\n");
		
	}

	fclose(f);
}


int main(int,char**) {
	test_forward();
	test_learning();
}
