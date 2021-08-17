# Feedforward Closedloop Learning

![alt tag](1.png)
![alt tag](2.png)

[Forward propagation closed loop learning
Bernd Porr, Paul Miller. Adaptive Behaviour 2019.](https://journals.sagepub.com/doi/10.1177/1059712319851070)

http://www.glasgowneuro.tech/fcl/

## Prerequisites

Ubuntu xenial or bionic LTS with swig installed.


## How to compile / install?

### From source (C++ and Python)
```
      cmake .
      make
      sudo make install
      sudo ./setup.py install
```

### From PyPi (Python only)

https://pypi.org/project/feedforward_closedloop_learning/

## Tests

These are in `tests_c` and `tests_py` to demonstrate both the C++ API and the python
API.

## Demos
The following demos are in a separate repository:

   * A classic line follower demo and
   * our vizdoom demo where our FCL agent fights against another automated agent
   
https://github.com/glasgowneuro/fcl_demos

## Class reference

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
