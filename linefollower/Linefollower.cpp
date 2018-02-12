#include <Enki.h>
#include "Racer.h"
#include <QApplication>
#include <QtGui>
#include "deep_feedback_learning.h"
#include <Iir.h>

#define IIRORDER 2

using namespace Enki;
using namespace std;

// size of the playground
double	maxx = 300;
double	maxy = 300;

#define STEPS_BELOW_ERR_THRESHOLD 2500

#define MAX_STEPS 100000

class LineFollower : public EnkiWidget {
protected:
	// The robot
	Racer* racer;
	
	const double speed = 90;
	const double fbgain = 300;

	int nInputs = 30;
	// We have one output neuron
	int nOutputs = 6;
	// We have two hidden layers
	int nHiddenLayers = 2;
	// We set two neurons in the first hidden layer
	int nNeuronsInHiddenLayers[6] = {9,6,6,6,6,6};
	// We set nFilters in the input
	int nFiltersInput = 10;
	// We set nFilters in the hidden unit
	int nFiltersHidden = 0;
	// Filterbank
	double minT = 2;
	double maxT = 30;

	double learningRate = 0.0001;
	
	DeepFeedbackLearning* deep_fbl = NULL;

	double* pred = NULL;
	double* err = NULL;

	FILE* flog = NULL;

	FILE* llog = NULL;

	Iir::Bessel::LowPass<IIRORDER> p0;
	Iir::Bessel::LowPass<IIRORDER> s0;

	double IRthres = 100;

	int learningOff = 1;

	double a = -0.5;

	double border = 25;

	long step = 0;

	double avgError = 0;
	double avgErrorDecay = 0.001;

	int successCtr = 0;
		
public:
	LineFollower(World *world, QWidget *parent = 0) :
		EnkiWidget(world, parent) {

		flog = fopen("log.dat","wt");
		llog = fopen("l.dat","wt");

		// setting up the robot
		racer = new Racer(nInputs);
		racer->pos = Point(100, 40);
		racer->angle = 1;
		racer->leftSpeed = speed;
		racer->rightSpeed = speed;
		world->addObject(racer);

		pred = new double[nInputs];
		err = new double[nNeuronsInHiddenLayers[0]];

		// setting up deep feedback learning
		deep_fbl = new DeepFeedbackLearning(
			nInputs,
			nNeuronsInHiddenLayers,
			nHiddenLayers,
			nOutputs,
			nFiltersInput,
			nFiltersHidden,
			minT,
			maxT);

		deep_fbl->initWeights(1,0,Neuron::MAX_OUTPUT_RANDOM);
		deep_fbl->setLearningRate(learningRate);
		deep_fbl->setLearningRateDiscountFactor(1);
		deep_fbl->setBias(1);
		deep_fbl->setUseDerivative(0);
		deep_fbl->setActivationFunction(Neuron::TANH);
		deep_fbl->setMomentum(0.9);
		
		p0.setup(IIRORDER,1,0.02);
		s0.setup(IIRORDER,1,0.05);

	}

	~LineFollower() {
		fclose(flog);
		fclose(llog);
		delete deep_fbl;
		delete[] pred;
		delete[] err;
	}

	void setLearningRate(double _learningRate) {
		if (_learningRate < 0) return;
		learningRate = _learningRate;
		deep_fbl->setLearningRate(learningRate);	
	}

	inline long getStep() {return step;}

	inline double getAvgError() {return avgError;}

	// here we do all the behavioural computations
	// as an example: line following and obstacle avoidance
	virtual void sceneCompletedHook()
	{
		double leftGround = racer->groundSensorLeft.getValue();
		double rightGround = racer->groundSensorRight.getValue();
		double leftGround2 = racer->groundSensorLeft2.getValue();
		double rightGround2 = racer->groundSensorRight2.getValue();

		fprintf(stderr,"%f ",racer->pos.x);
		// check if we've bumped into a wall
		if ((racer->pos.x<75) ||
		    (racer->pos.x>(maxx-border)) ||
		    (racer->pos.y<border) ||
		    (racer->pos.y>(maxy+border)) ||
		    (leftGround<a) ||
		    (rightGround<a) ||
		    (leftGround2<a) ||
		    (rightGround2<a)) {
			learningOff = 30;
		}
		if (racer->pos.x < border) {
			racer->angle = 0;
		}
		fprintf(stderr,"%d ",learningOff);
		if (learningOff>0) {
			deep_fbl->setLearningRate(0);
			learningOff--;
		} else {
			deep_fbl->setLearningRate(learningRate);
		}

		fprintf(stderr,"%f %f %f %f ",leftGround,rightGround,leftGround2,rightGround2);
		for(int i=0;i<racer->getNsensors();i++) {
			pred[i] = -(racer->getSensorArrayValue(i))*10;
			// workaround of a bug in Enki
			if (pred[i]<0) pred[i] = 0;
			//if (i>=racer->getNsensors()/2) fprintf(stderr,"%f ",pred[i]);
		}
		double error = (leftGround+leftGround2*2)-(rightGround+rightGround2*2);
       		avgError = avgError + (error - avgError)*avgErrorDecay;
		for(int i=0;i<nNeuronsInHiddenLayers[0];i++) {
			err[i] = error;
                }
		double sqAvgError = avgError * avgError;
		if (sqAvgError > 3E-7) {
			successCtr = 0;
		} else {
			successCtr++;
		}
		if (successCtr>STEPS_BELOW_ERR_THRESHOLD) {
			qApp->quit();
		}
		if (step>MAX_STEPS) {
			qApp->quit();
		}
		deep_fbl->doStep(pred,err);
		float vL = (deep_fbl->getOutputLayer()->getNeuron(0)->getOutput())*50 +
			(deep_fbl->getOutputLayer()->getNeuron(1)->getOutput())*10 +
			(deep_fbl->getOutputLayer()->getNeuron(2)->getOutput())*2;
		float vR = (deep_fbl->getOutputLayer()->getNeuron(3)->getOutput())*50 +
			(deep_fbl->getOutputLayer()->getNeuron(4)->getOutput())*10 +
			(deep_fbl->getOutputLayer()->getNeuron(5)->getOutput())*2;
		
		error = error * fbgain;
		fprintf(stderr,"%f ",error);
		fprintf(stderr,"%f ",vL);
		fprintf(stderr,"%f ",vR);
		fprintf(stderr,"\n");
		racer->leftSpeed = speed+error+vL;
		racer->rightSpeed = speed-error+vR;
		
		if (learningOff) error = 0;
		fprintf(flog,"%f %f %f ",error,vL,vR);
		for(int i=0;i<deep_fbl->getNumLayers();i++) {
			fprintf(flog,"%f ",deep_fbl->getLayer(i)->getWeightDistanceFromInitialWeights());
		}
		fprintf(flog,"\n");
		int n = 0;
		fprintf(llog,"%e ",avgError);
		fprintf(llog,"%e ",sqAvgError);
		for(int i=0;i<deep_fbl->getLayer(0)->getNeuron(0)->getNfilters();i++) {
			fprintf(llog,"%e ",deep_fbl->getLayer(0)->getNeuron(0)->getFilterOutput(n,i));
			fprintf(llog,"%e ",deep_fbl->getLayer(0)->getNeuron(0)->getWeight(n,i));
		}
		fprintf(llog,"%f\n",deep_fbl->getLayer(0)->getNeuron(0)->getOutput());
		if ((step%100)==0) {
			for(int i=0;i<deep_fbl->getNumLayers();i++) {
				char tmp[256];
				sprintf(tmp,"layer%d.dat",i);
				deep_fbl->getLayer(i)->saveWeightMatrix(tmp);
			}
		}
		step++;
	}

};

void singleRun(int argc,
	       char *argv[],
	       float learningrate = 0,
	       FILE* f = NULL) {
	QApplication app(argc, argv);
	QString filename("loop.png");
	QImage loopImage;
	loopImage = QGLWidget::convertToGLFormat(QImage(filename));
	if (loopImage.isNull()) {
		fprintf(stderr,"Racetrack file not found\n");
		exit(1);
	}
	const uint32_t *bitmap = (const uint32_t*)loopImage.constBits();
	World world(maxx, maxy,
		    Color(1000, 1000, 100),
		    World::GroundTexture(loopImage.width(), loopImage.height(), bitmap));
	LineFollower linefollower(&world);
	if (learningrate>0) {
		linefollower.setLearningRate(learningrate);
	}
	linefollower.show();
	app.exec();
	fprintf(stderr,"Finished.\n");
	if (f) {
		fprintf(f,"%e %ld %e\n",learningrate,linefollower.getStep(),linefollower.getAvgError());
	}
}

void statsRun(int argc,
	      char *argv[]) {
	FILE* f = fopen("stats.dat","wt");
	for(double learningRate = 0.0001; learningRate < 0.1; learningRate = learningRate * 2) {
		singleRun(argc,argv,learningRate,f);
		fflush(f);
	}
	fclose(f);
}


int main(int argc, char *argv[]) {
	int n = 0;
	if (argc>1) {
		n = atoi(argv[1]);
	} else {
		fprintf(stderr,"Single run: %s 0\n",argv[0]);
		fprintf(stderr,"Stats run: %s 1\n",argv[0]);
		return 0;
	}
	switch (n) {
	case 0:
		singleRun(argc,argv);
		break;
	case 1:
		statsRun(argc,argv);
		break;
	}
	return 0;
}
