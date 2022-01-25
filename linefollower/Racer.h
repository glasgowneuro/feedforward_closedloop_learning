#ifndef __ENKI_RACER_H
#define __ENKI_RACER_H

#include <enki/robots/DifferentialWheeled.h>
#include <enki/interactions/IRSensor.h>
#include <enki/interactions/CircularCam.h>
#include <enki/interactions/GroundSensor.h>

namespace Enki
{
	class Racer : public DifferentialWheeled
	{
	public:
		Racer(int _nSensors,
		      int _sensorArrayPos1 = 12, double _sensorArrayWidth1 = 10,
		      int _sensorArrayPos2 = 20, double _sensorArrayWidth2 = 20	);

		inline int getNsensors() {
			return nSensors;
		}

		inline double getSensorArrayValue(int index) {
			return groundSensorArray[index]->getValue();
		}

		int nSensors;

		int sensorArrayPos1;
		double sensorArrayWidth1;
		
		int sensorArrayPos2;
		double sensorArrayWidth2;
		
		IRSensor infraredSensorLeft;
		IRSensor infraredSensorRight;
		GroundSensor groundSensorLeft;
		GroundSensor groundSensorRight;
		GroundSensor groundSensorLeft2;
		GroundSensor groundSensorRight2;

		GroundSensor** groundSensorArray;
		
	};
}

#endif

