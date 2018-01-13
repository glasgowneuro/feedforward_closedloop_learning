#include "Racer.h"

namespace Enki
{
	Racer::Racer(int _nSensors,
		     int _sensorArrayPos1, double _sensorArrayWidth1,
		     int _sensorArrayPos2, double _sensorArrayWidth2
		) :
		DifferentialWheeled(5.2, 100, 0.05),
		infraredSensorLeft(this, Vector(10, 5),  1.8, M_PI/4, 10, 1200, -0.9, 7, 20),
		infraredSensorRight(this, Vector(10, -5), 1.8, -M_PI/4,10, 1200, -0.9, 7, 20),
		groundSensorLeft (this, Vector(10, 10), 0, 1, 1, 0),
		groundSensorRight(this, Vector(10, -10),0, 1, 1, 0),
		groundSensorLeft2 (this, Vector(10, 15), 0, 1, 1, 0),
		groundSensorRight2(this, Vector(10, -15),0, 1, 1, 0)
	{
		nSensors = _nSensors;
		if (nSensors%2) {
			fprintf(stderr,"BUG: nSensors needs to be even.\n");
		}

		sensorArrayPos1 = _sensorArrayPos1;
		sensorArrayWidth1 = _sensorArrayWidth1;

		sensorArrayPos2 = _sensorArrayPos2;
		sensorArrayWidth2 = _sensorArrayWidth2;

		addLocalInteraction(&infraredSensorLeft);
		addLocalInteraction(&infraredSensorRight);
		addLocalInteraction(&groundSensorLeft);
		addLocalInteraction(&groundSensorRight);
		addLocalInteraction(&groundSensorLeft2);
		addLocalInteraction(&groundSensorRight2);

		groundSensorArray = new GroundSensor*[nSensors];
		
		double d1 = sensorArrayWidth1*2 / ((nSensors/2)-1);
		double d2 = sensorArrayWidth2*2 / ((nSensors/2)-1);
		
		for(int i=0;i<(nSensors/2);i++) {
			float y1 = -sensorArrayWidth1+d1*i;
			float y2 = -sensorArrayWidth2+d2*i;
			fprintf(stderr,"sensor %d pos = %f,%f\n",i,y1,y2);
			groundSensorArray[i*2] = new GroundSensor(this, Vector(sensorArrayPos1,y1), 0, 1, 1, -0.731059);
			groundSensorArray[i*2+1] = new GroundSensor(this, Vector(sensorArrayPos2,y2), 0, 1, 1, -0.731059);
			addLocalInteraction(groundSensorArray[i*2]);
			addLocalInteraction(groundSensorArray[i*2+1]);
		}
		
		setRectangular(20,10,5, 80);
	}
}
