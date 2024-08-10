#ifndef LINEFOLLOWER_H
#define LINEFOLLOWER_H

const double speed = 90;
const double fbgain = 250;

const int nInputs = 30;

// Number of layers of neurons in total
static constexpr int nLayers = 3;

// The number of neurons in every layer
const std::vector<int> nNeuronsInLayers = {20,12,6};
// const std::vector<int> nNeuronsInLayers = {1,1};


// The number of neurons in every layer for model prediction
const std::vector<int> nNeuronsInLayersBackProp = {10,9,9,1};

// We set nFilters in the input
// const int nFiltersInput = 10;
const int nFiltersInput = 5;


// We set nFilters in the unit
const int nFilters = 0;

// Filterbank
const double minT = 2;
const double maxT = 20; // 30

// size of the playground
double	maxx = 300;
double	maxy = 300;

// for stats
#define SQ_ERROR_THRES 0.001
#define STEPS_BELOW_ERR_THRESHOLD 1000

// max number of steps to terminate
#define MAX_STEPS 15000 //15000

// terminates if the agent won't turn after these steps
#define STEPS_OFF_TRACK 1000

const double a = -0.5;

const double border = 25;

const double avgErrorDecay = 0.01;

const double IRthres = 100;

#endif
