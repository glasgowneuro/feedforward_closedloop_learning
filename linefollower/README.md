# FCL Linefollower

The line follower demonstrates as a simple scenario how
FCL works. The robot has a default steering mechanism
to track the line and this generates the error signal for FCL.

FCL then learns with the help of this error signal to
improve its behaviour. The inputs to FCL are two rows
of sensors in front of the robot.

## Pre-requisites

  - Ubuntu Linux LTS

  - QT5 development libraries with openGL and GLU

  - ENKI: https://github.com/glasgowneuro/enki
    Install with `cmake .` -- `make` -- `sudo make install`.

## Compilation

`cmake .` and `make` to compile it.

## Running the line follower

The line follower has two modes: single run or stats run.
In the single run mode it runs until the squared average of the
error signal is below a certain threshold (SQ_ERROR_THRES).
In the stats run it performs a logarithmic sweep of different
learning rates and counts the simulation steps till success.

## Data logging

There are two log files: `flog.tsv` and `llog.tsv`. The
data is space separated and every time step has one row.

### flog.dat

This log records the steering actions of the robot:

`amplified_error steering_left steering_right`

### llog.dat

The error signal can be seen as the performance measure
of learning and it slowly decays to zero which is logged here:

`unamplified_error average_error absolute_error`

The script `plot_abs_error.py` plots the error signal while
the line follower is running.

### Weights

Run the script `plotweights.py` which plots the weights while
the line follower is running.
