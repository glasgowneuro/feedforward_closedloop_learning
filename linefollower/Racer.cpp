#include "Racer.h"

namespace Enki
{
	Racer::Racer() :
		DifferentialWheeled(5.2, 100, 0.05),
		infraredSensorLeft(this, Vector(10, 5),  1.8, M_PI/4, 10, 1200, -0.9, 7, 20),
		infraredSensorRight(this, Vector(10, -5), 1.8, -M_PI/4,10, 1200, -0.9, 7, 20),
		groundSensorLeft (this, Vector(10, 5), 0, 1, 1, 0),
		groundSensorRight(this, Vector(10, -5),0, 1, 1, 0)
	{
		addLocalInteraction(&infraredSensorLeft);
		addLocalInteraction(&infraredSensorRight);
		addLocalInteraction(&groundSensorLeft);
		addLocalInteraction(&groundSensorRight);

		groundSensorArray = new GroundSensor*[nSensors];
		double width = 20;
		double d = width*2 / (nSensors-1);
		for(int i=0;i<nSensors;i++) {
			groundSensorArray[i] = new GroundSensor(this, Vector(sensorArrayPos, -width+d*i), 0, 1, 1, -0.731059);
			addLocalInteraction(groundSensorArray[i]);
		}
		
		setRectangular(20,10,5, 80);
	}
}
