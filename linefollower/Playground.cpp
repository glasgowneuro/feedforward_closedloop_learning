#include <Enki.h>
#include "Racer.h"
#include <QApplication>
#include <QtGui>

using namespace Enki;
using namespace std;

class EnkiPlayground : public EnkiWidget
{
protected:
	Racer* racer;

	const double speed = 20;
	
public:
	EnkiPlayground(World *world, QWidget *parent = 0) :
		EnkiWidget(world, parent)
	{
		racer = new Racer;
		racer->pos = Point(40, 60);
		racer->leftSpeed = speed;
		racer->rightSpeed = speed;
		world->addObject(racer);
	}

	// here we do all the behavioural computations
	// as an example: line following and obstacle avoidance
	virtual void sceneCompletedHook()
	{
		double leftIR = racer->infraredSensor1.getValue();
		double rightIR = racer->infraredSensor4.getValue();
		//fprintf(stderr,"%f %f\n",left,right);
		double leftGround = racer->groundSensorLeft.getValue();
		double rightGround = racer->groundSensorRight.getValue();
		double error = leftGround-rightGround;
		fprintf(stderr,"%f %f %f\n",leftGround,rightGround,error);
		double gain = 100;
		racer->leftSpeed = speed-rightIR/100+error*gain;
		racer->rightSpeed = speed-leftIR/100-error*gain;		
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
	EnkiPlayground viewer(&world);
	
	viewer.show();
	
	return app.exec();
}
