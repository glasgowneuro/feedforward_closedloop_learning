#include "deep_feedback_learning.h"
#include <Iir.h>
#include<stdio.h>

void test_forward() {
	int nFiltersInput = 10;
	int nFiltersHidden = 10;
	int nHidden[] = {2};
	
	DeepFeedbackLearning* deep_fbl = new DeepFeedbackLearning(2,nHidden,1,nFiltersInput,nFiltersHidden,100,200);
	FILE* f=fopen("test_deep_fbl_cpp_forward.dat","wt");
	deep_fbl->setLearningRate(0.0);
	
	double input[2];
	double error[2];

	for(int i=0;i<nFiltersInput;i++) {
		deep_fbl->getLayer(0)->getNeuron(0)->setWeight(0,i,1);
	}


	for(int i=0; i<deep_fbl->getNumHidLayers(); i++) {
		for(int j=0;j<nFiltersHidden;j++) {
			deep_fbl->getLayer(i+1)->getNeuron(0)->setWeight(0,j,1);
		}
	}

	for(int n = 0; n < 100;n++) {

		input[0] = 0;
		if ((n>10)&&(n<20)) {
			input[0] = 0.001;
		}
		fprintf(f,"%f ",input[0]);

		deep_fbl->doStep(input,error);
		for(int i=0; i<deep_fbl->getNumHidLayers(); i++) {
			fprintf(f,"%f ",deep_fbl->getLayer(i)->getNeuron(0)->getSum());
		}
		fprintf(f,"%f ",deep_fbl->getOutputLayer()->getNeuron(0)->getOutput());
		
		fprintf(f,"\n");
		
	}

	fclose(f);
}



void test_learning() {
	int nHidden[] = {2};
	DeepFeedbackLearning* deep_fbl = new DeepFeedbackLearning(2,nHidden,1);
	deep_fbl->initWeights(0.01);
	
	FILE* f=fopen("test_deep_fbl_cpp_learning.dat","wt");

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

		deep_fbl->doStep(input,error);

		for(int i=0;i<2;i++) {
			for(int j=0;j<2;j++) {
				for(int k=0; k<deep_fbl->getNumHidLayers(); k++) {
					fprintf(f, "%f ",
							deep_fbl->getLayer(k)->getNeuron(i)->getWeight(j));
				}
			}
		}
		for(int i=0;i<1;i++) {
			for(int j=0;j<2;j++) {
				fprintf(f,
					"%f ",
					deep_fbl->getOutputLayer()->getNeuron(i)->getWeight(j));
			}
		}
		for(int i=0;i<1;i++) {
			fprintf(f,
				"%f ",
				deep_fbl->getOutputLayer()->getNeuron(i)->getOutput());
		}
		fprintf(f,"\n");
		
	}

	fclose(f);
}

void test_learning_and_filters() {
	int nInputs = 2;
	int nHidden[] = {2};
	int nOutput = 1;
	int nFiltersPerInput = 2;
	int nFiltersPerHidden = 2;
	double min_filter_time = 100;
	double max_filter_time = 200;
	DeepFeedbackLearning* deep_fbl = new DeepFeedbackLearning(nInputs,nHidden,nOutput,
					  nFiltersPerInput,nFiltersPerHidden,
					  min_filter_time,max_filter_time);
	deep_fbl->initWeights(0.01);
	deep_fbl->setLearningRate(0.1);
	
	FILE* f=fopen("test_deep_fbl_cpp_learning_with_filters.dat","wt");

	double input[2];
	double error[2];	
	
	for(int n = 0; n < 1000;n++) {
		
		double stim = 0;
		double err = 0;
		if ((n>100)&&(n<200)) {
			stim = 1;
			if ((n>190)&&(n<200)) {
				err = 1;
			}
			if ((n>250)&&(n<300)) {
				err = -1;
			}
		}
		fprintf(f,"%e %e ",stim,err);

		input[0] = stim;
		error[0] = err;

		deep_fbl->doStep(input,error);

		for(int i=0;i<2;i++) {
			for(int j=0;j<2;j++) {
				for(int k=0; k<deep_fbl->getNumHidLayers(); k++) {
					fprintf(f,
							"%e ",
							deep_fbl->getLayer(k)->getNeuron(i)->getWeight(j));
				}
			}
		}
		for(int i=0;i<1;i++) {
			for(int j=0;j<2;j++) {
				fprintf(f,
					"%e ",
					deep_fbl->getOutputLayer()->getNeuron(i)->getWeight(j));
			}
		}
		for(int i=0;i<1;i++) {
			fprintf(f,
				"%e ",
				deep_fbl->getOutputLayer()->getNeuron(i)->getOutput());
		}
		fprintf(f,"\n");
		
	}

	fclose(f);
}


int main(int,char**) {
	test_forward();
	test_learning();
	test_learning_and_filters();
}
