#ifndef STARS_CUH
#define STARS_CUH

#include "structs.cuh"

Stars* allocateMemoryToStars(int numStars, int numLambdas);
void deallocateStars(Stars* stars);
Stars* starsTransferToDevice(Stars* h_stars, int numLambdas);
Stars* readStars(FrequenciesData* freqData);
void calculateCumulativeSpec(Stars* stars, FrequenciesData* freqData);
void calculateStarsLuminosities(Stars* stars, FrequenciesData* freqData);
void calculateEnergyStars(Stars* stars, int numPhotons);
void jitterStars(Stars* stars, Grid* grid);
Stars* setUpStars(FrequenciesData* freqData, Grid* grid, int numPhotons);
#endif
