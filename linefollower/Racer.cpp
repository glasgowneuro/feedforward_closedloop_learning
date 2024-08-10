#include "Racer.h"

namespace Enki
{
	Racer::Racer(int _nSensors,
		     int _sensorArrayPos1, double _sensorArrayWidth1,
		     int _sensorArrayPos2, double _sensorArrayWidth2
		) :
		DifferentialWheeled(8, 100, 0.05),
		infraredSensorLeft(this, Vector(11, 10), 100, M_PI/200, 2, 1200, -0.9, 7, 20),
		infraredSensorRight(this, Vector(11, -10), 100, -M_PI/200, 2, 1200, -0.9, 7, 20),
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
		
		float d1 = (float)(sensorArrayWidth1*2 / ((nSensors/2)-1));
		float d2 = (float)(sensorArrayWidth2*2 / ((nSensors/2)-1));
		
		for(int i=0;i<(nSensors/2);i++) {
			float y1 = (float)(-sensorArrayWidth1+d1*(float)i);
			float y2 = (float)(-sensorArrayWidth2+d2*(float)i);
			fprintf(stderr,"sensor %d pos = %f,%f\n",i,y1,y2);
			groundSensorArray[i] = new GroundSensor(this, Vector(sensorArrayPos1,y1), 0, 1, 1, -0.731059);
			groundSensorArray[i+nSensors/2] = new GroundSensor(this, Vector(sensorArrayPos2,y2), 0, 1, 1, -0.731059,2);
			addLocalInteraction(groundSensorArray[i]);
			addLocalInteraction(groundSensorArray[i+nSensors/2]);
		}
		
		setRectangular(20,10,5,80);
				// setRectangular(30,15,5,80);

	}
}
