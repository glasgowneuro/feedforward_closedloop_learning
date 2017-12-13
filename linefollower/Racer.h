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
		IRSensor infraredSensorLeft;
		IRSensor infraredSensorRight;
		GroundSensor groundSensorLeft;
		GroundSensor groundSensorRight;
		GroundSensor** groundSensorArray;
		
	public:
		Racer();
		const int nSensors = 5;
	};
}

#endif

