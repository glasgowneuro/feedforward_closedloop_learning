#include "Racer.h"
#include <QApplication>
#include <QtGui>
#include "fcl_util.h"
#include <viewer/Viewer.h>
#include <vector>

#include "Linefollower.h"

using namespace Enki;
using namespace std;

class LineFollower : public ViewerWidget {
protected:
	// The robot
	Racer* racer;
	
	// Is set by the learning rate setter. Do not change here!
	double learningRate = 0;
	
	FeedforwardClosedloopLearningWithFilterbank* fcl = NULL;

	std::vector<double> pred;
	std::vector<double> err;


	FILE* flog = NULL;

	FILE* llog = NULL;

	FILE* fcoord = NULL;
	
	int learningOff = 1;

	long step = 0;
	long stepBeforeSuccess = 0;

	double avgError = 0;
	double avgErrorAfterLearning = 0;

	int successCtr = 0;

	int trackCompletedCtr = 5000;
	
	// Flag to represent the finish of training.
	int trainingFinished = 0;

	// Sliding window average.
	double avgOutput[6] = {0};

	double weightChange = 0;
	double weightSum = 0;

	// Identify goodness of previous data.
	double nowDistance = 0;
	double lastDistance = 0;

	// history step = 3.
	// double lastError[3] = {0};
	double lastControl = 0.0f;

	double lastError = 0.0f;

	double lastVL = 0.0f;

	// NN control error input.
	std::vector<double> nnError;

	const int delayNum = 10;

	bool isLearning = true;
	bool isCounting = false;

public:
	LineFollower(World *world, QWidget *parent = 0) :
		ViewerWidget(world, parent) {

		flog = fopen("flog.tsv","wt");
		fcoord = fopen("coord.tsv","wt");

		// setting up the robot
		racer = new Racer(nInputs);
		racer->pos = Point(100, 40);
		// racer->pos = Point(100, 10);
		racer->angle = 1;
		racer->leftSpeed = speed;
		racer->rightSpeed = speed;
		world->addObject(racer);

		pred.resize(nInputs);
		err.resize(nInputs);
		nnError.resize(nInputs);

		// setting up deep feedforward learning
		fcl = new FeedforwardClosedloopLearningWithFilterbank(
			nInputs,
			nNeuronsInLayers,
			nFiltersInput,
			minT,
			maxT);

		fcl->initWeights(1,0,FCLNeuron::MAX_OUTPUT_RANDOM);
		fcl->setLearningRate(learningRate);
		fcl->setLearningRateDiscountFactor(1);
		fcl->setBias(1);
		fcl->setActivationFunction(FCLNeuron::TANH);
		fcl->getOutputLayer()->setActivationFunction(FCLNeuron::ActivationFunction::LINEAR);
		fcl->setMomentum(0.9);	
	}

	~LineFollower() {
		fclose(flog);
		fclose(fcoord);
		delete fcl;
	}

	void setLearningRate(double _learningRate) {
		if (_learningRate < 0) return;
		learningRate = _learningRate;
		fcl->setLearningRate(learningRate);
	}

	inline long getStep() {return step;}

	inline long getStepToConverge() {return (1000 + (step - successCtr));}

	inline double getAvgError() {return avgError;}

	inline double getAvgErrorAfterLearning() {return avgErrorAfterLearning;}

	inline double getWeightChange() {return weightChange;}

	inline bool getTrainingFinished() {return isLearning;}

	// here we do all the behavioural computations
	// as an example: line following and obstacle avoidance
	virtual void sceneCompletedHook()
	{
		// These 4 variables are for reflex loop.  
		double leftGround = racer->groundSensorLeft.getValue();
		double rightGround = racer->groundSensorRight.getValue();
		double leftGround2 = racer->groundSensorLeft2.getValue();
		double rightGround2 = racer->groundSensorRight2.getValue();
		
		// fprintf(stderr,"%e\t",racer->pos.x);
		fprintf(fcoord,"%e\t%e\n",racer->pos.x,racer->pos.y);
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
			
			// learningOff = 5;
		}

		if ((racer->pos.x<75) && (racer->pos.y<150) && isLearning == false && isCounting == false) 
		{
			isCounting = true;
		}

		if (racer->pos.x < border) {
			racer->angle = 0;
			trackCompletedCtr = STEPS_OFF_TRACK;
		}
		trackCompletedCtr--;
		if (trackCompletedCtr < 1) {
			// been off the track for a long time!
			step = MAX_STEPS;
			qApp->quit();
		}
		//fprintf(stderr,"%d ",learningOff);
		if (learningOff>0) {
			fcl->setLearningRate(0);
			learningOff--;
		} else {
			fcl->setLearningRate(learningRate);
		}

		fprintf(stderr,"%e %e %e %e ",leftGround,rightGround,leftGround2,rightGround2);
		for(int i=0;i<racer->getNsensors();i++) {
			pred[i] = -(racer->getSensorArrayValue(i))*2;
			// workaround of a bug in Enki
			if (pred[i]<0) pred[i] = 0;
			//if (i>=racer->getNsensors()/2) fprintf(stderr,"%e ",pred[i]);
		}

		/* When the robot is on track, sensors capture a large value, vise versa. */ 
		double error = (leftGround+leftGround2*2)-(rightGround+rightGround2*2);
		fprintf(stderr, "%e ", error);

		// no need to weight with the left steering!
		double errorToNetwork = error; // * fabs(lastVL) / (fabs(lastVL) + fabs(lastControl) + 0.00001);

		//fprintf(stderr,"%e %e %e %e ",pred[0], pred[14], pred[15], pred[29]);

		for(auto &e:err) {
			e = errorToNetwork;
                }

		fcl->doStep(pred,err);
   		/****************************************************/

		float vL = (float)((fcl->getOutputLayer()->getNeuron(0)->getOutput())*100 +
							(fcl->getOutputLayer()->getNeuron(1)->getOutput())*20 +
							(fcl->getOutputLayer()->getNeuron(2)->getOutput())*5);
		float vR = (float)((fcl->getOutputLayer()->getNeuron(3)->getOutput())*100 +
							(fcl->getOutputLayer()->getNeuron(4)->getOutput())*20 +
							(fcl->getOutputLayer()->getNeuron(5)->getOutput())*5);

		if(vL > 50) vL = 50;
		if(vR > 50) vR = 50;

		double erroramp = error * fbgain;

		fprintf(stderr, "%ld ", step);
		fprintf(stderr, "%d ", successCtr);
		fprintf(stderr,"\n");

		racer->leftSpeed = speed + erroramp + vL;
		racer->rightSpeed = speed - (erroramp + vR);

		weightChange = fcl->getLayer(0)->getWeightDistanceFromInitialWeights();
		
		// documenting
		// if the learning is off we set the error to zero which
		// happens on the edges when the robot is turned violently around
		if (learningOff) error = 0;
		avgError = avgError + (error - avgError)*avgErrorDecay;
		double absError = fabs(avgError);

		/* Record the avgerror in the next 1000 steps after successful learning. */
		if (isCounting == true) 
		{
			avgErrorAfterLearning = avgErrorAfterLearning + (error - avgErrorAfterLearning)*avgErrorDecay;
			double absErrorAfterLearning = fabs(avgErrorAfterLearning);
			fprintf(stderr, "After learning! step:%ld, avgErr:%e \n", stepBeforeSuccess, absErrorAfterLearning);
		}

		if (absError > SQ_ERROR_THRES) {
			successCtr = 0;
		} else {
			successCtr++;
		}

		if (stepBeforeSuccess > 1000) {
			qApp->quit();
		}

		// if (successCtr>STEPS_BELOW_ERR_THRESHOLD) {
		// 	qApp->quit();
		// }	

		// if (step>MAX_STEPS) {
		// 		qApp->quit();
		// 	}	

		if (step>MAX_STEPS && isLearning == true) {
			qApp->quit();
		}

		if(vL == 50) {
			qApp->quit();
		}

		/********************************************************************************/
		/* Printing! */
		// The derivative of activation function shown below:
		// for (int i = 0; i < fcl->getOutputLayer()->getNneurons(); i++)
		// {
		// 	fprintf(flog, "%e\t", dActivation[i]);
		// }

		// fprintf(flog,"%e\t",error);
		// fprintf(flog,"%e\t",erroramp);
		// fprintf(flog,"%e\t%e",vL,vR); 
		// fprintf(flog,"\t%e", racer->leftSpeed - racer->rightSpeed);

		// for (int i = 0; i < fcl->getLayer(1)->getNneurons(); i++)
		// {
		// 	fprintf(flog, "\t%e", fcl->getLayer(1)->getError(i));
		// }
				
		// for (int i = 0; i < fcl->getLayer(1)->getNneurons(); i++)
		// {
		// 	fprintf(flog, "\t%e", fcl->getLayer(1)->getOutput(i));
		// }

		fprintf(flog, "\t%e", fcl->getOutputLayer()->getError(0));
		fprintf(flog, "\t%e", erroramp/fbgain);


		fprintf(flog, "\t%e", fcl->getOutputLayer()->getOutput(0));

		fprintf(flog, "\t%e", fcl->getLayer(0)->getWeightDistanceFromInitialWeights());
		fprintf(flog, "\t%e", fcl->getLayer(1)->getWeightDistanceFromInitialWeights());

		fprintf(flog, "\t%e", fcl->getOutputLayer()->getWeightDistanceFromInitialWeights());

		fprintf(flog,"\n");
		/********************************************************************************/

		if ((step%100)==0) {
			for(int i=0;i<fcl->getNumLayers();i++) {
				char tmp[256];
				sprintf(tmp,"layer%d.dat",i);
				fcl->getLayer(i)->saveWeightMatrix(tmp);
			}
		}

		lastError = error;

		lastVL = vL;

		lastControl = erroramp;

		step++;
		if (isCounting == true) stepBeforeSuccess++;

		// if (step==11000) {
		// 	for(int i=0;i<fcl->getNumLayers();i++) {
		// 		char tmp[256];
		// 		sprintf(tmp,"layer%d.dat",i);
		// 		fcl->getLayer(i)->saveWeightMatrix(tmp);
		// 	}
		// }
	}

};


void singleRun(int argc,
	       char *argv[],
	       float learningrate,
	       FILE* f = NULL) {
	QApplication app(argc, argv);
	// QString filename("loop.png");
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
	linefollower.setLearningRate(learningrate);
	linefollower.show();
	app.exec();
	fprintf(stderr,"Finished.\n");
	if (f) {
		fprintf(f,"%e %ld %e %e %d\n",learningrate,linefollower.getStepToConverge(),linefollower.getAvgError(), linefollower.getAvgErrorAfterLearning(), linefollower.getTrainingFinished());
	}
}


void statsRun(int argc,
	      char *argv[]) {
	FILE* f = fopen("stats.dat","wt");
	for(float learningRate = 3.814697e-03f; learningRate < 1; learningRate = learningRate * 1.25f) {
		// srandom(1);
		// singleRun(argc,argv,learningRate,f);
		// fflush(f);
		srandom(42);
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
		singleRun(argc,argv ,0.0025f);
		break;
	case 1:
		statsRun(argc,argv);
		break;
	case 2:
		singleRun(argc,argv,0);
		break;
	}
	return 0;
}      
