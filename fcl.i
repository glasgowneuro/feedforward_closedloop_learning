%module(docstring="Feedforward Closedloop Learning") feedforward_closedloop_learning
%{
	#define SWIG_FILE_WITH_INIT
	#include "fcl.h"
	#include "fcl/layer.h"
	#include "fcl/neuron.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}

%feature("autodoc", "3");

%apply (double* IN_ARRAY1, int DIM1) {(double* input, int n1), (double* error, int n2)};
%apply (int* IN_ARRAY1, int DIM1) {(int* num_of_hidden_neurons_per_layer_array, int _num_hid_layers)};

%include "fcl.h"
%include "fcl/layer.h"
%include "fcl/neuron.h"
