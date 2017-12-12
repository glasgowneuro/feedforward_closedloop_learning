#include <Enki.h>
#include "Racer.h"
#include <QApplication>
#include <QtGui>
#include "deep_feedback_learning.h"

using namespace Enki;
using namespace std;

class LineFollower : public EnkiWidget
{
protected:
	Racer* racer;

	const double speed = 30;
	const double fbgain = 100;

	// number of sensor array inputs
	int nInputs = 1;
	// We have one output neuron
	int nOutputs = 1;
	// We have two hidden layers
	int nHiddenLayers = 3;
	// We set two neurons in the first hidden layer
	int nNeuronsInHiddenLayers[4] = {2,2,2,2};
	// We set nFilters in the input
	int nFiltersInput = 10;
	// We set nFilters in the hidden unit
	int nFiltersHidden = 10;
	// Filterbank
	double minT = 100;
	double maxT = 500;
	
	DeepFeedbackLearning* deep_fbl = NULL;

	double* pred = NULL;
	double* err = NULL;

public:
	LineFollower(World *world, QWidget *parent = 0) :
		EnkiWidget(world, parent) {
		
		// setting up the robot
		racer = new Racer;
		racer->pos = Point(40, 60);
		racer->leftSpeed = speed;
		racer->rightSpeed = speed;
		world->addObject(racer);

		nInputs = racer->nSensors;
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
		deep_fbl->setLearningRate(0.01);
		deep_fbl->setLearningRateDiscountFactor(1);
		deep_fbl->setAlgorithm(DeepFeedbackLearning::ico);
		deep_fbl->setBias(1);
		deep_fbl->setUseDerivative(1);
		
	}

	// here we do all the behavioural computations
	// as an example: line following and obstacle avoidance
	virtual void sceneCompletedHook()
	{
		double leftIR = racer->infraredSensorLeft.getValue();
		double rightIR = racer->infraredSensorRight.getValue();
		//fprintf(stderr,"%f %f\n",left,right);
		double leftGround = racer->groundSensorLeft.getValue();
		double rightGround = racer->groundSensorRight.getValue();
		double error = leftGround-rightGround;
		//fprintf(stderr,"%f %f %f\n",leftGround,rightGround,error);
		for(int i=0;i<racer->nSensors;i++) {
		}
		for(int i=0;i<nInputs;i++) {
			pred[i] = -(racer->groundSensorArray[i]->getValue())*100;
			fprintf(stderr,"%f ",pred[i]);
		}
		for(int i=0;i<nNeuronsInHiddenLayers[0];i++) {
                        err[i] = error;
                }
		deep_fbl->doStep(pred,err);
		float v = deep_fbl->getOutputLayer()->getNeuron(0)->getOutput();
		v = v * 10;
		fprintf(stderr,"%f ",v);
		fprintf(stderr,"\n");
		racer->leftSpeed = speed-rightIR/100+error*fbgain+v;
		racer->rightSpeed = speed-leftIR/100-error*fbgain-v;
	}

};

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);

	QString filename("track.png");
	QImage gt;
	gt = QGLWidget::convertToGLFormat(QImage(filename));
	if (gt.isNull()) {
		fprintf(stderr,"Texture file not found\n");
		exit(1);
	}
	const uint32_t *bits = (const uint32_t*)gt.constBits();
	World world(120, Color(0.9, 0.9, 0.9), World::GroundTexture(gt.width(), gt.height(), bits));
	LineFollower linefollower(&world);
	linefollower.show();
	return app.exec();
}
