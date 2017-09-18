CFLAGS = -g -O0 -march=native -std=c++11 -I./..
LDFLAGS = -pthread -std=c++11 -liir ../deep_feedback_learning.a

all: test_deep_feedback_learning test_neuron test_closed_loop

test_deep_feedback_learning.o: test_deep_feedback_learning.cpp
	g++ $(CFLAGS) -c test_deep_feedback_learning.cpp

test_deep_feedback_learning: ../deep_feedback_learning.a test_deep_feedback_learning.o
	g++ -o test_deep_feedback_learning test_deep_feedback_learning.o $(LDFLAGS) 

test_closed_loop.o: test_closed_loop.cpp
	g++ $(CFLAGS) -c test_closed_loop.cpp

test_closed_loop: ../deep_feedback_learning.a test_closed_loop.o
	g++ -o test_closed_loop test_closed_loop.o $(LDFLAGS) 

test_neuron.o: test_neuron.cpp
	g++ $(CFLAGS) -c test_neuron.cpp

test_neuron: test_neuron.o
	g++ -o test_neuron test_neuron.o $(LDFLAGS) 


clean:
	rm -rf *.o test_deep_feedback_learning test_closed_loop test_neuron *~ *.dat *.csv *.dat
