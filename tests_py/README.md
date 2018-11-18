# test_dfl_learning_with_filters.py

Stress test for the Python API.

We do learning in a network with one hidden layer and
bandpass filters at both the input and the hidden
layer.

The stimulus is sent in between timestep 100..105 and
the error from time steps 105..110.

This is repeated then every 200 time steps and then plotted.
