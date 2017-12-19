#ifndef __ENKI_RACER_H
#define __ENKI_RACER_H

#include <DifferentialWheeled.h>
#include <IRSensor.h>
#include <CircularCam.h>
#include <GroundSensor.h>

namespace Enki
{
	class Racer : public DifferentialWheeled
	{
	public:
		Racer(int _nSensors = 5, int _sensorArrayPos = 50, double _sensorArrayWidth = 20);

		inline int getNsensors() {
			return nSensors;
		}

		inline double getSensorArrayValue(int index) {
			return groundSensorArray[index]->getValue();
		}

		int sensorArrayPos;
		double sensorArrayWidth;
		int nSensors;
		
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

