%module deep_ico
%{
	#define SWIG_FILE_WITH_INIT
	#include "deep_ico.h"
	#include "layer.h"
	#include "neuron.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (float* IN_ARRAY1, int DIM1) {(float* data1, int n1), (float* data2, int n2)};

%include "deep_ico.h"
%include "layer.h"
%include "neuron.h"
