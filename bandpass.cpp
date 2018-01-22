#include "bandpass.h"
#include <math.h>
#include <complex>
#include <stdio.h>
#ifdef __linux__
#include <unistd.h>
#endif


Bandpass::Bandpass() {
	reset();
	norm=1;
}


void Bandpass::reset() {
	buffer0=0;
	buffer1=0;
	buffer2=0;
	actualOutput=0;
}	


void Bandpass::setParameters(double f,double q) {
	std::complex<double> s1;
	std::complex<double> s2;
	
	assert(q>0.5);
	assert(f<=0.5);
	assert(f>0);
	double fTimesPi=f*M_PI*2;
	double e=fTimesPi/(2.0*q);
	assert((fTimesPi*fTimesPi-e*e)>=0);
	double w=sqrt(fTimesPi*fTimesPi-e*e);
	s1=std::complex<double>(-e,w);
	s2=std::complex<double>(-e,-w);
	enumerator0=1;
	enumerator1=0;
	enumerator2=0;
	denominator0=1;
	denominator1=real(-exp(s2)-exp(s1));
	denominator2=real(exp(s1+s2));
	calcNorm(f);
}


void Bandpass::impulse(char* name) {
  int steps=100;
#ifdef DEBUG_BP
  fprintf(stderr,"Impulse resp: %s, %d steps\n",name,steps);
#endif
  for(int i=0;i<steps;i++) {
    filter(0);
  }
  double input=0.0;
  FILE* ff=fopen(name,"wt");
  if (!ff) {
    fprintf(stderr,"Couldn't open %s \n",name);
    return;
  }
  for(int i=0;i<steps;i++) {
    if (i==10) {
      input=1.0F;
    } else {
      input=0.0F;
    }
    fprintf(ff,"%d %f\n",i,filter(input));
  }
  fclose(ff);
}


double Bandpass::filter(double value) {
  double input=0.0;
  double output=0.0;
  // a little bit cryptic but a little bit optimized for speed
  input=value;
  output=(enumerator1*buffer1);
  input=input-(denominator1*buffer1);
  output=output+(enumerator2*buffer2);
  input=input-(denominator2*buffer2);
  output=output+input*enumerator0;
  buffer2=buffer1;
  buffer1=input;
  output=output/norm;
  actualOutput=output;
  return output;
}


void Bandpass::calcNorm(double f) {
	double max = 0;
	norm = 1;
	for(int i=0;i<(2/f);i++) {
		double v = 0;
		if (i>1) {
			v = 1;
		}
		double v2 = fabs(filter(v));
		if (v2>max) max = v2;
	}
//	fprintf(stderr,"BPmax=%f\n",max);
//	sleep(1);
	norm = max;
}


void Bandpass::calcPolesZeros(double f,double r) {
//	fprintf(stderr,"Bandpass: f=%f,r=%f\n",f,r);
	enumerator0=1;
	enumerator1=0;
	enumerator2=0;
	denominator0=1;
	denominator1=-2*r*cos(2*M_PI*f);
	denominator2=r*r;
	calcNorm(f);
}
