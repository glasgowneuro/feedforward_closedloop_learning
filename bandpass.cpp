#include "bandpass.h"
#include <math.h>
#include <complex>
#include <stdio.h>


Bandpass::Bandpass() {
  buffer[0]=0;
  buffer[1]=0;
  buffer[2]=0;
  actualOutput=0;
  norm=1;
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
  output=(enumerator[1]*buffer[1]);
  input=input-(denominator[1]*buffer[1]);
  output=output+(enumerator[2]*buffer[2]);
  input=input-(denominator[2]*buffer[2]);
  output=output+input*enumerator[0];
  buffer[2]=buffer[1];
  buffer[1]=input;
  output=output/norm;
  actualOutput=output;
  return output;
}




void Bandpass::calcPolesZeros(double f,double r) {
//	fprintf(stderr,"Bandpass: f=%f,r=%f\n",f,r);
	enumerator[0]=1;
	enumerator[1]=0;
	enumerator[2]=0;
	denominator[0]=1;
	denominator[1]=-2*r*cos(2*M_PI*f);
	denominator[2]=r*r;

	float max = 0;
	for(int i=0;i<(1/f*3);i++) {
		float v = 0;
		if (i==((int)(1/f))) {
			v = 10;
		}
		float v2 = filter(v);
		if (v2>max) max = v2;
	}
	norm = max;
}
