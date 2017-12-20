#include <Enki.h>
#include "Racer.h"
#include <QApplication>
#include <QtGui>
#include "deep_feedback_learning.h"
#include <Iir.h>

#define IIRORDER 2

using namespace Enki;
using namespace std;

class LineFollower : public EnkiWidget
{
protected:
	Racer* racer;

	const double speed = 90;
	const double fbgain = 300;

	int nInputs = 5;
	// We have one output neuron
	int nOutputs = 2;
	// We have two hidden layers
	int nHiddenLayers = 3;
	// We set two neurons in the first hidden layer
	int nNeuronsInHiddenLayers[6] = {5,5,5,5,5,5};
	// We set nFilters in the input
	int nFiltersInput = 10;
	// We set nFilters in the hidden unit
	int nFiltersHidden = 0;
	// Filterbank
	double minT = 5;
	double maxT = 50;

	double learningRate = 0.1;
	
	DeepFeedbackLearning* deep_fbl = NULL;

	double* pred = NULL;
	double* err = NULL;

	FILE * flog = NULL;

	Iir::Bessel::LowPass<IIRORDER> p0;
	Iir::Bessel::LowPass<IIRORDER> s0;

public:
	LineFollower(World *world, QWidget *parent = 0) :
		EnkiWidget(world, parent) {

		flog = fopen("log.dat","wt");
		
		// setting up the robot
		racer = new Racer(nInputs,50);
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
		deep_fbl->setAlgorithm(DeepFeedbackLearning::ico);
		deep_fbl->setBias(0);
		deep_fbl->setUseDerivative(0);
	
		p0.setup(IIRORDER,1,0.02);
		s0.setup(IIRORDER,1,0.05);
	}

	~LineFollower() {
		fclose(flog);
	}

	// here we do all the behavioural computations
	// as an example: line following and obstacle avoidance
	virtual void sceneCompletedHook()
	{
		double leftIR = racer->infraredSensorLeft.getValue();
		double rightIR = racer->infraredSensorRight.getValue();
		//fprintf(stderr,"%f %f\n",leftIR,rightIR);
		if ((leftIR>50) || (rightIR>50)) {
			deep_fbl->setLearningRate(0);
		} else {
			deep_fbl->setLearningRate(learningRate);
		}
		if (leftIR<100) leftIR = 0;
		if (rightIR<100) rightIR = 0;
		double leftGround = racer->groundSensorLeft.getValue();
		double rightGround = racer->groundSensorRight.getValue();
		double leftGround2 = racer->groundSensorLeft2.getValue();
		double rightGround2 = racer->groundSensorRight2.getValue();
		double error = (leftGround+leftGround2*2)-(rightGround+rightGround2*2);
		//fprintf(stderr,"%f %f %f\n",leftGround,rightGround,error);
		for(int i=0;i<racer->getNsensors();i++) {
		}
		for(int i=0;i<nInputs;i++) {
			pred[i] = -(racer->getSensorArrayValue(i))*10;
			fprintf(stderr,"%f ",pred[i]);
		}
		for(int i=0;i<nNeuronsInHiddenLayers[0];i++) {
                        err[i] = error;
                }
		deep_fbl->doStep(pred,err);
		float vL = (deep_fbl->getOutputLayer()->getNeuron(0)->getOutput())*100;
		float vR = (deep_fbl->getOutputLayer()->getNeuron(1)->getOutput())*100;
		error = error * fbgain;
		fprintf(stderr,"%f ",error);
		fprintf(stderr,"%f ",vL);
		fprintf(stderr,"%f ",vR);
		fprintf(stderr,"\n");
		double dreflex = p0.filter(rightIR-leftIR);
		double sreflex = s0.filter(rightIR+leftIR);
		racer->leftSpeed = speed+dreflex/4+error+vL-sreflex;
		racer->rightSpeed = speed-dreflex/4-error+vR-sreflex;
		fprintf(flog,"%f %f %f ",error,vL,vR);
		for(int i=0;i<deep_fbl->getNumLayers();i++) {
			fprintf(flog,"%f ",deep_fbl->getLayer(i)->getWeightDistanceFromInitialWeights());
		}
		fprintf(flog,"\n");
	}

};

int main(int argc, char *argv[])
{
	QApplication app(argc, argv);

	QString filename("loop.png");
	QImage loopImage;
	loopImage = QGLWidget::convertToGLFormat(QImage(filename));
	if (loopImage.isNull()) {
		fprintf(stderr,"Racetrack file not found\n");
		exit(1);
	}
	const uint32_t *bitmap = (const uint32_t*)loopImage.constBits();
	World world(300, 300,
		    Color(0.9, 0.9, 0.9),
		    World::GroundTexture(loopImage.width(), loopImage.height(), bitmap));
	LineFollower linefollower(&world);
	linefollower.show();
	return app.exec();
}
