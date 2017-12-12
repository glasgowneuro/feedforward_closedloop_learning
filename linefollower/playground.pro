TEMPLATE = app
TARGET = playground
INCLUDEPATH += .
INCLUDEPATH += /usr/local/include/enki

# Input
HEADERS = Racer.h
SOURCES += Playground.cpp Racer.cpp

QT += opengl widgets
CONFIG          += qt warn_on debug
QMAKE_CXXFLAGS += -std=c++0x -march=native

LIBS	+= /usr/local/lib/libenki.a
