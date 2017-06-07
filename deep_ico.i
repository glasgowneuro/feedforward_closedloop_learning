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

%apply (double* IN_ARRAY1, int DIM1) {(double* input, int n1), (double* error, int n2)};

%include "deep_ico.h"
%include "layer.h"
%include "neuron.h"
