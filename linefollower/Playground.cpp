/*
    Enki - a fast 2D robot simulator
    Copyright (C) 2017 Bernd Porr <mail@berndporr.me.uk>
    Copyright (C) 1999-2016 Stephane Magnenat <stephane at magnenat dot net>
    Copyright (C) 2004-2005 Markus Waibel <markus dot waibel at epfl dot ch>
    Copyright (c) 2004-2005 Antoine Beyeler <abeyeler at ab-ware dot com>
    Copyright (C) 2005-2006 Laboratory of Intelligent Systems, EPFL, Lausanne
    Copyright (C) 2006-2008 Laboratory of Robotics Systems, EPFL, Lausanne
    See AUTHORS for details

    This program is free software; the authors of any publication 
    arising from research using this software are asked to add the 
    following reference:
    Enki - a fast 2D robot simulator
    http://home.gna.org/enki
    Stephane Magnenat <stephane at magnenat dot net>,
    Markus Waibel <markus dot waibel at epfl dot ch>
    Laboratory of Intelligent Systems, EPFL, Lausanne.

    You can redistribute this program and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

/*
Custom Robot example which has ground sensors and follows a line
It has also a camera which looks to the front and IR sensors
*/

#include <Enki.h>
#include "Racer.h"
#include <QApplication>
#include <QtGui>
#include <iostream>

using namespace Enki;
using namespace std;

class EnkiPlayground : public EnkiWidget
{
protected:
	QMap<PhysicalObject*, int> bullets;

	Racer* racer;

	const double speed = 20;
	
public:
	EnkiPlayground(World *world, QWidget *parent = 0) :
		EnkiWidget(world, parent)
	{
		#define PROBLEM_GENERIC_TOY
		#define PROBLEM_BALL_LINE
		
		#ifdef PROBLEM_GENERIC_TOY
		{
			const double amount = 9;
			const double radius = 5;
			const double height = 20;
			Enki::Polygon p;
			for (double a = 0; a < 2*M_PI; a += 2*M_PI/amount)
				p.push_back(Point(radius * cos(a), radius * sin(a)));
			
			PhysicalObject* o = new PhysicalObject;
			PhysicalObject::Hull hull(Enki::PhysicalObject::Part(p, height));
			o->setCustomHull(hull, -1);
			o->setColor(Color(0.4,0.6,0.8));
			o->pos = Point(100, 100);
			world->addObject(o);
		}
		
		for (int i = 0; i < 10; i++)
		{
			PhysicalObject* o = new PhysicalObject;
			o->pos = Point(UniformRand(20, 100)(), UniformRand(20, 100)());
			o->setCylindric(1, 1, 10);
			o->setColor(Color(0.9, 0.2, 0.2));
			o->dryFrictionCoefficient = 0.01;
			world->addObject(o);
		}
		
		Enki::Polygon p2;
		p2.push_back(Point(5,1));
		p2.push_back(Point(-5,1));
		p2.push_back(Point(-5,-1));
		p2.push_back(Point(5,-1));
		for (int i = 0; i < 5; i++)
		{
			PhysicalObject* o = new PhysicalObject;
			PhysicalObject::Hull hull(Enki::PhysicalObject::Part(p2, 3));
			o->setCustomHull(hull, 30);
			o->setColor(Color(0.2, 0.1, 0.6));
			o->collisionElasticity = 0.2;
			o->pos = Point(UniformRand(20, 100)(), UniformRand(20, 100)());
			world->addObject(o);
		}
		
		// cross shape
		{
			PhysicalObject* o = new PhysicalObject;
			PhysicalObject::Hull hull;
			hull.push_back(Enki::PhysicalObject::Part(Enki::Polygon() << Point(5,1) << Point(-5,1) << Point(-5,-1) << Point(5,-1), 0.1));
			hull.push_back(Enki::PhysicalObject::Part(Enki::Polygon() << Point(1,5) << Point(-1,5) << Point(-1,-5) << Point(1,-5), 0.1));
			o->setCustomHull(hull, 60);
			o->setColor(Color(0.2, 0.4, 0.6));
			o->collisionElasticity = 0.2;
			o->pos = Point(UniformRand(20, 100)(), UniformRand(20, 100)());
			world->addObject(o);
		}
		#endif // PROBLEM_GENERIC_TOY
		
		#ifdef PROBLEM_BALL_LINE
		for (double d = 40; d < 60; d += 8)
		{
			PhysicalObject* o = new PhysicalObject;
			o->pos = Point(d, 20);
			o->setCylindric(4, 2, 10);
			o->setColor(Color(0.2, 0.2, 0.6));
			o->dryFrictionCoefficient = 0.;
			world->addObject(o);
		}
		#endif // PROBLEM_BALL_LINE
		
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

