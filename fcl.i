%module(docstring="Feedforward Closedloop Learning") feedforward_closedloop_learning
%{
	#define SWIG_FILE_WITH_INIT
	#include "fcl.h"
	#include "fcl_util.h"
	#include "fcl/layer.h"
	#include "fcl/neuron.h"
%}

%include exception.i

%exception {
    try {
        $action
    } catch (const char* e) {
        PyErr_SetString(PyExc_RuntimeError, e);
        return NULL;
    }
}

%include <typemaps.i>
%include "std_vector.i"

%template(DoubleVector) std::vector<double>;
%template(IntVector) std::vector<int>;

%feature("autodoc", "3");



%include "fcl.h"
%include "fcl_util.h"
%include "fcl/layer.h"
%include "fcl/neuron.h"
