# Feedforward Closedloop Learning (FCL)

[Forward propagation closed loop learning
Bernd Porr, Paul Miller. Adaptive Behaviour 2019.](https://journals.sagepub.com/doi/10.1177/1059712319851070)

[Submission version](https://www.berndporr.me.uk/Porr_Miller_FCL_2019_Adaptive_Behaviour.pdf)

## Error _forward_ propagation

![alt tag](2.png)
![alt tag](1.png)

For an autonomous agent, the inputs are the sensory data that inform the agent of the state of the world, and the outputs are their actions, which act on the world and consequently produce new sensory inputs. The agent only knows of its own actions via their effect on future inputs; therefore desired states, and error signals, are most naturally defined in terms of the inputs. Most machine learning algorithms, however, operate in terms of desired outputs. For example, backpropagation takes target output values and propagates the corresponding error backwards through the network in order to change the weights. In closed loop settings, it is far more obvious how to define desired sensory inputs than desired actions, however. To train a deep network using errors defined in the input space would call for an algorithm that can propagate those errors _forwards_ through the network, from input layer to output layer, in much the same way that activations are propagated.

## Prerequisites

Ubuntu LTS with swig installed.


## How to compile / install?

### From source (C++ and Python)
```
      cmake .
      make
      sudo make install
      ./setup.py install --user
```

### From PyPi (Python only)

https://pypi.org/project/feedforward_closedloop_learning/

## Tests

These are in `tests_c` and `tests_py` to demonstrate both the C++ API and the python
API.

## Demos

   * A classic line follower demo in `linefollower/` and
   * our vizdoom demo where our FCL agent fights against another automated agent: https://github.com/glasgowneuro/fcl_doom

## Class reference

The online documentation can be found here: https://glasgowneuro.github.io/feedforward_closedloop_learning/

Run `doxygen` to generate the reference documentation for all classes:
```
make docs
```
This will be written to the `docs` subdirectory in HTML, RTF and LaTeX.

## License

GNU GENERAL PUBLIC LICENSE

Version 3, 29 June 2007

```
(C) 2017,2018, Bernd Porr <bernd@glasgowneuro.tech>
(C) 2017,2018, Paul Miller <paul@glasgowneuro.tech>
```
## Citation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1944837.svg)](https://doi.org/10.5281/zenodo.1944837)
