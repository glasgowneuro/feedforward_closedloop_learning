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
		Racer(int _nSensors = 10,
		      int _sensorArrayPos1 = 50, double _sensorArrayWidth1 = 20,
		      int _sensorArrayPos2 = 100, double _sensorArrayWidth2 = 30	);

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

