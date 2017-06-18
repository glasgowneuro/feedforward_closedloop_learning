%module deep_feedback_learning
%{
	#define SWIG_FILE_WITH_INIT
	#include "deep_feedback_learning.h"
	#include "layer.h"
	#include "neuron.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%apply (double* IN_ARRAY1, int DIM1) {(double* input, int n1), (double* error, int n2)};

%include "deep_feedback_learning.h"
%include "layer.h"
%include "neuron.h"
