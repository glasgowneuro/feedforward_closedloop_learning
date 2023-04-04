#include "Racer.h"
#include <QApplication>
#include <QtGui>
#include "cldl_filterbank.h"
#include <viewer/Viewer.h>

using namespace Enki;
using namespace std;

#include "../Linefollower.h"

class LineFollower : public ViewerWidget {
protected:
	// The robot
	Racer* racer;
	
	// Is set by the learning rate setter. Do not change here!
	double learningRate = 0;
	
	ClosedloopDeepLearningWithFilterbank* cldl = NULL;

	double* pred = NULL;
	double* err = NULL;

	FILE* flog = NULL;

	FILE* fcoord = NULL;

	int learningOff = 1;

	long step = 0;

	double avgError = 0.0;

	int successCtr = 0;

	int trackCompletedCtr = 5000;
		
public:
	LineFollower(World *world, QWidget *parent = 0) :
		ViewerWidget(world, parent) {

		flog = fopen("flog.tsv","wt");
		fcoord = fopen("coord.tsv","wt");

		// setting up the robot
		racer = new Racer(nInputs);
		racer->pos = Point(100, 40);
		racer->angle = 1;
		racer->leftSpeed = speed;
		racer->rightSpeed = speed;
		world->addObject(racer);

		pred = new double[nInputs];
		err = new double[nNeuronsInLayers[2]];

		// setting up deep feedforward learning
		cldl = new ClosedloopDeepLearningWithFilterbank(
			nInputs,
			nNeuronsInLayers.data(),
			(int)nNeuronsInLayers.size(),
			nFiltersInput,
			minT,
			maxT);

		cldl->initNetwork(CLDLNeuron::W_RANDOM_NORM, CLDLNeuron::B_NONE, CLDLNeuron::Act_Tanh);
		cldl->setLearningRate(learningRate);
	}

	~LineFollower() {
		fclose(flog);
		fclose(fcoord);
		delete cldl;
		delete[] pred;
		delete[] err;
	}

	void setLearningRate(double _learningRate) {
		if (_learningRate < 0) return;
		learningRate = _learningRate;
		cldl->setLearningRate(learningRate);	
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

		fprintf(stderr,"%e\t",racer->pos.x);
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
		fprintf(stderr,"%d ",learningOff);
		if (learningOff>0) {
			cldl->setLearningRate(0);
			learningOff--;
		} else {
			cldl->setLearningRate(learningRate);
		}

		fprintf(stderr,"%e %e %e %e ",leftGround,rightGround,leftGround2,rightGround2);
		for(int i=0;i<racer->getNsensors();i++) {
			pred[i] = -(racer->getSensorArrayValue(i))*10;
			// workaround of a bug in Enki
			if (pred[i]<0) pred[i] = 0;
			//if (i>=racer->getNsensors()/2) fprintf(stderr,"%e ",pred[i]);
		}
		double error = (leftGround+leftGround2*2)-(rightGround+rightGround2*2);
		for(int i=0;i<nNeuronsInLayers[0];i++) {
			err[i] = error;
                }
		// !!!!
		cldl->doStep(pred,err);
		float vL = (float)((cldl->getOutput(0))*50 +
				   (cldl->getOutput(1))*10 +
				   (cldl->getOutput(2))*2);
		float vR = (float)((cldl->getOutput(3))*50 +
				   (cldl->getOutput(4))*10 +
				   (cldl->getOutput(5))*2);
		
		double erroramp = error * fbgain;
		fprintf(stderr,"%e ",erroramp);
		fprintf(stderr,"%e ",vL);
		fprintf(stderr,"%e ",vR);
		fprintf(stderr,"\n");
		racer->leftSpeed = speed+erroramp+vL;
		racer->rightSpeed = speed-erroramp+vR;

		// documenting
		// if the learning is off we set the error to zero which
		// happens on the edges when the robot is turned violently around
		if (learningOff) error = 0;
       		avgError = avgError + (error - avgError)*avgErrorDecay;
		double absError = fabs(avgError);
		if (absError > SQ_ERROR_THRES) {
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
		
		fprintf(flog,"%e\t",error);
		fprintf(flog,"%e\t",avgError);
		fprintf(flog,"%e\t%e",vL,vR);
		for(int i=0;i<cldl->getnLayers();i++) {
			fprintf(flog,"\t%e",cldl->getLayer(i)->getWeightDistance());
		}
		fprintf(flog,"\n");

		fflush(flog);
		if ((step%100)==0) {
			for(int i=0;i<cldl->getnLayers();i++) {
				char tmp[256];
				sprintf(tmp,"layer%d.dat",i);
				cldl->getLayer(i)->snapWeights(tmp);
			}
		}

		step++;
	}

};


void singleRun(int argc,
	       char *argv[],
	       float learningrate,
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
	linefollower.setLearningRate(learningrate);
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
	for(float learningRate = 0.00001f; learningRate < 0.1; learningRate = learningRate * 1.25f) {
		srandom(1);
		singleRun(argc,argv,learningRate,f);
		fflush(f);
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
		singleRun(argc,argv,0.025f);
		break;
	case 1:
		statsRun(argc,argv);
		break;
	}
	return 0;
}
