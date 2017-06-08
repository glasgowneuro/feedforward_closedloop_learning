CFLAGS = -g -O2 -march=native

all: test_neuron test_deep_ico py_module

test: test_neuron test_deep_ico
	./test_neuron > test.dat
	./test_deep_ico

test_neuron: neuron.o test_neuron.cpp
	g++ $(CFLAGS) test_neuron.cpp neuron.o -o test_neuron

test_deep_ico: neuron.o layer.o deep_ico.o test_deep_ico.o
	g++ test_deep_ico.o neuron.o layer.o deep_ico.o -o test_deep_ico $(LDFLAGS) 

test_deep_ico.o: test_deep_ico.cpp
	g++ $(CFLAGS) -c test_deep_ico.cpp

neuron.o: neuron.cpp neuron.h
	g++ -fPIC $(CFLAGS) -c neuron.cpp

layer.o: layer.cpp layer.h
	g++ -fPIC $(CFLAGS) -c layer.cpp

deep_ico.o: deep_ico.cpp deep_ico.h
	g++ $(CFLAGS) -fPIC -c deep_ico.cpp

clean:
	rm -rf *.o test_deep_ico test_neuron *~ *.dat deep_ico_wrap.cxx *.so *.pyc

py_module: deep_ico.i 
	swig -c++ -python deep_ico.i
	g++ -fPIC $(CFLAGS) -c deep_ico_wrap.cxx -I /usr/include/python3.5
	g++ -fPIC -shared neuron.o layer.o deep_ico.o deep_ico_wrap.o -o _deep_ico.so
