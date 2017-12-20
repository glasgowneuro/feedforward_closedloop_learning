#include "Racer.h"

namespace Enki
{
	Racer::Racer(int _nSensors, int _sensorArrayPos, double _sensorArrayWidth) :
		DifferentialWheeled(5.2, 100, 0.05),
		infraredSensorLeft(this, Vector(10, 5),  1.8, M_PI/4, 10, 1200, -0.9, 7, 20),
		infraredSensorRight(this, Vector(10, -5), 1.8, -M_PI/4,10, 1200, -0.9, 7, 20),
		groundSensorLeft (this, Vector(10, 7.5), 0, 1, 1, 0),
		groundSensorRight(this, Vector(10, -7.5),0, 1, 1, 0),
		groundSensorLeft2 (this, Vector(10, 10), 0, 1, 1, 0),
		groundSensorRight2(this, Vector(10, -10),0, 1, 1, 0)
	{
		nSensors = _nSensors;
		sensorArrayPos = _sensorArrayPos;
		sensorArrayWidth = _sensorArrayWidth;

		addLocalInteraction(&infraredSensorLeft);
		addLocalInteraction(&infraredSensorRight);
		addLocalInteraction(&groundSensorLeft);
		addLocalInteraction(&groundSensorRight);
		addLocalInteraction(&groundSensorLeft2);
		addLocalInteraction(&groundSensorRight2);

		groundSensorArray = new GroundSensor*[nSensors];
		double d = sensorArrayWidth*2 / (nSensors-1);
		for(int i=0;i<nSensors;i++) {
			float y = -sensorArrayWidth+d*i;
			fprintf(stderr,"sensor %d pos = %f\n",i,y);
			groundSensorArray[i] = new GroundSensor(this, Vector(sensorArrayPos,y), 0, 1, 1, -0.731059);
			addLocalInteraction(groundSensorArray[i]);
		}
		
		setRectangular(20,10,5, 80);
	}
}
