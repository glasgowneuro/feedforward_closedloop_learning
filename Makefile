CFLAGS = -g -O0 -march=native -std=c++11 -DRANGE_CHECKS
#CFLAGS = -Ofast -march=native -std=c++11
LDFLAGS = -pthread -std=c++11 -liir

all: deep_feedback_learning.py deep_feedback_learning.a tests_c
	ulimit -c unlimited

tests_c: deep_feedback_learning.a
	make -C tests_c

bandpass.o: bandpass.cpp bandpass.h globals.h
	g++ -fPIC $(CFLAGS) -c bandpass.cpp

neuron.o: neuron.cpp neuron.h bandpass.o globals.h
	g++ -fPIC $(CFLAGS) -c neuron.cpp

layer.o: layer.cpp layer.h neuron.o bandpass.o globals.h 
	g++ -fPIC $(CFLAGS) -c layer.cpp

deep_feedback_learning.o: deep_feedback_learning.cpp deep_feedback_learning.h layer.o neuron.o bandpass.o globals.h
	g++ $(CFLAGS) -fPIC -c deep_feedback_learning.cpp

deep_feedback_learning.a: bandpass.o neuron.o layer.o deep_feedback_learning.o
	ar rcs deep_feedback_learning.a bandpass.o neuron.o layer.o deep_feedback_learning.o

deep_feedback_learning.py: deep_feedback_learning.i bandpass.o neuron.o layer.o deep_feedback_learning.o
	swig -c++ -python -py3 deep_feedback_learning.i
	g++ -fPIC $(CFLAGS) -c deep_feedback_learning_wrap.cxx -I /usr/include/python3.5
	g++ -fPIC -shared bandpass.o neuron.o layer.o deep_feedback_learning.o deep_feedback_learning_wrap.o -o _deep_feedback_learning.so $(LDFLAGS) 

clean:
	rm -rf *.o test_deep_feedback_learning test_neuron *~ *.dat deep_feedback_learning_wrap.cxx *.so *.pyc deep_feedback_learning.py *.csv *.dat
	make -C tests_c clean
