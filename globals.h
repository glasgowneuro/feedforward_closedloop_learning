#ifndef GLOBALS_H
#define GLOBALS_H

// This disables the assert statement
// The define RANGE_CHECKS is defined in the Makefile
// for debugging purposes together with the
// optimisation flags

#ifndef RANGE_CHECKS
#define NDEBUG
#endif

// danger zone
// usually commented out

// no derivative of the activation function
// #define NO_DERIV_ACTIVATION



#ifdef _WIN32

// that includes the standard M_PI etc stuff
#define _USE_MATH_DEFINES
#include <cmath>  

#endif



#endif
