CFLAGS = -g -Ofast -march=native -std=c++11
LDFLAGS = -pthread -std=c++11 -liir

all: test_neuron test_deep_feedback_learning deep_feedback_learning.py

test: test_neuron test_deep_feedback_learning
	./test_neuron > test.dat
	./test_deep_feedback_learning

test_neuron: neuron.o test_neuron.cpp bandpass.o
	g++ $(CFLAGS) test_neuron.cpp neuron.o bandpass.o -o test_neuron $(LDFLAGS) 

test_deep_feedback_learning: bandpass.o neuron.o layer.o deep_feedback_learning.o test_deep_feedback_learning.o
	g++ test_deep_feedback_learning.o neuron.o layer.o deep_feedback_learning.o bandpass.o -o test_deep_feedback_learning $(LDFLAGS) 

test_deep_feedback_learning.o: test_deep_feedback_learning.cpp
	g++ $(CFLAGS) -c test_deep_feedback_learning.cpp

bandpass.o: bandpass.cpp bandpass.h
	g++ -fPIC $(CFLAGS) -c bandpass.cpp

neuron.o: neuron.cpp neuron.h bandpass.o
	g++ -fPIC $(CFLAGS) -c neuron.cpp

layer.o: layer.cpp layer.h neuron.o bandpass.o
	g++ -fPIC $(CFLAGS) -c layer.cpp

deep_feedback_learning.o: deep_feedback_learning.cpp deep_feedback_learning.h layer.o neuron.o bandpass.o
	g++ $(CFLAGS) -fPIC -c deep_feedback_learning.cpp

clean:
	rm -rf *.o test_deep_feedback_learning test_neuron *~ *.dat deep_feedback_learning_wrap.cxx *.so *.pyc deep_feedback_learning.py *.csv *.dat

deep_feedback_learning.py: deep_feedback_learning.i bandpass.o neuron.o layer.o deep_feedback_learning.o
	swig -c++ -python -py3 deep_feedback_learning.i
	g++ -fPIC $(CFLAGS) -c deep_feedback_learning_wrap.cxx -I /usr/include/python3.5
	g++ -fPIC -shared bandpass.o neuron.o layer.o deep_feedback_learning.o deep_feedback_learning_wrap.o -o _deep_feedback_learning.so $(LDFLAGS) 
