# Feedforward Closedloop Learning

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
