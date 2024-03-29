Feedforward closed loop learning (FCL) is a learning algorithm
which adds flexibility to autonomous agents.

A designer defines an initial behaviour as a reflex and then FCL
learns from the reflex to develop new flexible behaviours.

The Python documentation can be obtained with::
  
    import feedforward_closedloop_learning as fcl
    help(fcl)

The Python API is identical to the C++ API: The documentation can be
found here:
https://glasgowneuro.github.io/feedforward_closedloop_learning/

The best way to get started is to look at the script
in `tests_py`:
https://github.com/glasgowneuro/feedforward_closedloop_learning/tree/master/tests_py

A full application using the Python API is our vizdoom
agent:
https://github.com/glasgowneuro/fcl_demos
