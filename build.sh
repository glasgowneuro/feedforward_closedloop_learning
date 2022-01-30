#!/bin/sh
rm -rf CMakeCache.txt CMakeFiles build dist
cmake .
make
