CFLAGS = -g -O0 -march=native -std=c++11 -I./..
LDFLAGS = -pthread -std=c++11 -liir ../deep_feedback_learning.a

all: test_deep_feedback_learning test_neuron test_layer

test_deep_feedback_learning.o: test_deep_feedback_learning.cpp 
	g++ $(CFLAGS) -c test_deep_feedback_learning.cpp

test_deep_feedback_learning: ../deep_feedback_learning.a test_deep_feedback_learning.o  ../deep_feedback_learning.a
	g++ -o test_deep_feedback_learning test_deep_feedback_learning.o $(LDFLAGS) 

test_neuron.o: test_neuron.cpp
	g++ $(CFLAGS) -c test_neuron.cpp

test_neuron: test_neuron.o  ../deep_feedback_learning.a
	g++ -o test_neuron test_neuron.o $(LDFLAGS) 

test_layer.o: test_layer.cpp
	g++ $(CFLAGS) -c test_layer.cpp

test_layer: test_layer.o  ../deep_feedback_learning.a
	g++ -o test_layer test_layer.o $(LDFLAGS) 

test_bandpass.o: test_bandpass.cpp
	g++ $(CFLAGS) -c test_bandpass.cpp

test_bandpass: test_bandpass.o  ../deep_feedback_learning.a
	g++ -o test_bandpass test_bandpass.o $(LDFLAGS) 


clean:
	rm -rf *.o test_deep_feedback_learning test_neuron test_bandpass *~ *.dat *.csv *.dat
