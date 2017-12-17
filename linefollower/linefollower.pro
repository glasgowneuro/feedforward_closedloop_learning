TEMPLATE = app
TARGET = linefollower
INCLUDEPATH += .
INCLUDEPATH += /usr/local/include/enki
INCLUDEPATH += ..

# Input
HEADERS = Racer.h
SOURCES += Linefollower.cpp Racer.cpp

QT += opengl widgets
CONFIG          += qt warn_on debug
QMAKE_CXXFLAGS += -std=c++0x -march=native

LIBS	+= /usr/local/lib/libenki.a
LIBS	+= ../deep_feedback_learning.a
LIBS	+= -liir
