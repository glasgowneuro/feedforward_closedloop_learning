#include "deep_feedback_learning.h"
#include <Iir.h>
#include<stdio.h>

// inits the network with random weights with quite a few hidden units so that
// a nice response is generated
void test_forward() {
	int nFiltersInput = 10;
	int nFiltersHidden = 10;
	int nHidden[] = {10,10};

	DeepFeedbackLearning* deep_fbl = new DeepFeedbackLearning(2,nHidden,2,1,nFiltersInput,nFiltersHidden,100,200);
	FILE* f=fopen("test_deep_fbl_cpp_forward.dat","wt");
	deep_fbl->setLearningRate(0.0);
	// random init
	deep_fbl->initWeights(0.1);

	double input[2];
	double error[2];

	for(int n = 0; n < 1000;n++) {

		input[0] = 0;
		if ((n>10)&&(n<20)) {
			input[0] = 0.1;
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
	DeepFeedbackLearning* deep_fbl = new DeepFeedbackLearning(2,nHidden,1,1);
	deep_fbl->initWeights(0.1,0,Neuron::MAX_OUTPUT_CONST);
	deep_fbl->setLearningRate(0.01);
	deep_fbl->setAlgorithm(DeepFeedbackLearning::ico);
	
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
			if ((n>700)&&(n<800)) {
				err = -1;
			}
		}
		fprintf(f,"%f %f ",stim,err);

		input[0] = stim;
		error[0] = err;
		error[1] = err;

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
	int nHidden[] = {2};
	int nFiltersInput = 5;
	int nFiltersHidden = 5;
	double minT = 3;
	double maxT = 15;
	
	DeepFeedbackLearning* deep_fbl = new DeepFeedbackLearning(2,nHidden,1,1,nFiltersInput, nFiltersHidden, minT,maxT);
	deep_fbl->initWeights(0.001,0,Neuron::MAX_OUTPUT_CONST);
	deep_fbl->setLearningRate(0.0001);
	deep_fbl->setAlgorithm(DeepFeedbackLearning::ico);
	deep_fbl->setBias(0);
	
	FILE* f=fopen("test_deep_fbl_cpp_learning_with_filters.dat","wt");

	double input[1];
	double error[2];

	int rep = 200;
	
	for(int n = 0; n < 10000;n++) {
		
		double stim = 0;
		double err = 0;
		if (((n%rep)>100)&&((n%rep)<105)) {
			stim = 1;
		}
		if (((n%rep)>105)&&((n%rep)<110)&&(n<9000)) {
			err = 1;
		}
		fprintf(f,"%e %e ",stim,err);

		input[0] = stim;
		error[0] = err;
		error[1] = err;

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



#define IIRORDER 2

void test_closedloop() {
	int nHidden[] = {2};
	int nFiltersInput = 5;
	int nFiltersHidden = 5;
	double minT = 3;
	double maxT = 15;
	
	DeepFeedbackLearning* deep_fbl = new DeepFeedbackLearning(2,nHidden,1,1,nFiltersInput, nFiltersHidden, minT,maxT);
	deep_fbl->initWeights(0.001,0,Neuron::MAX_OUTPUT_CONST);
	deep_fbl->setLearningRate(0.0001);
	deep_fbl->setAlgorithm(DeepFeedbackLearning::ico);
	deep_fbl->setBias(0);
	
	Iir::Bessel::LowPass<IIRORDER> p0;
	p0.setup (IIRORDER,1,0.1);
	
	Iir::Bessel::LowPass<IIRORDER> h0;
	h0.setup (IIRORDER,1,0.1);
	
	FILE* f=fopen("test_closedloop.dat","wt");

	double input[1];
	double error[2];

	float v = 0;
	float v0 = 0;
	float x0 = 0;
	float err = 0;
	float dist = 0;
	float pred = 0;
	
	for(int step = 0; step < 10000; step++) {

		int n = step % 1000;
		
		pred = 0;
		dist = 0;
		if ((n>100)&&(n<1000)) {
			pred = 1;
			if ((n>500)&&(n<800)) {
				dist = 1;
			}
			if ((n>700)&&(n<1000)) {
				dist = 0;
			}
		}

		input[0] = pred;
		error[0] = err;
		error[1] = err;
		
		deep_fbl->doStep(input,error);

		v0 = dist - v;

		v0 = v0 + deep_fbl->getOutputLayer()->getNeuron(0)->getOutput();

		x0 = p0.filter(v0);

		float fb_gain = 0.2;

		err = - x0 * fb_gain;
		
		v = h0.filter(err);
		
		fprintf(f,"%d %f %f %f %f %f ",n,pred,dist,err,x0,v0);

		fprintf(f,
			"%f ",
			deep_fbl->getOutputLayer()->getNeuron(0)->getOutput());

		for(int i=0;i<2;i++) {
			for(int j=0;j<2;j++) {
				fprintf(f,
					"%f ",
					deep_fbl->getLayer(0)->getNeuron(i)->getWeight(j));
			}
		}
		for(int i=0;i<2;i++) {
			for(int j=0;j<2;j++) {
				fprintf(f,
					"%f ",
					deep_fbl->getOutputLayer()->getNeuron(i)->getWeight(j));
			}
		}
		fprintf(f,"\n");
		
	}

	fclose(f);
}






int main(int,char**) {
	test_forward();
	test_learning();
	test_learning_and_filters();
	test_closedloop();
}
