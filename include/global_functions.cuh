#ifndef GLOBAL_FUNCTIONS_CUH
#define GLOBAL_FUNCTIONS_CUH
#include "structs.cuh"

SimulationParameters* parametersTransferToDevice(SimulationParameters h_params);
SimulationParameters readInputParameters(int argc, char **argv);
void checkSimulationParameters(SimulationParameters param);
/*
--------------------------------------------------------------------------
                THE BLACKBODY PLANCK FUNCTION B_nu(T)

     This function computes the Blackbody function

                    2 h nu^3 / c^2
        B_nu(T)  = ------------------    [ erg / cm^2 s ster Hz ]
                   exp(h nu / kT) - 1

     ARGUMENTS:
        freq    [Hz]            = Frequency
        temp  [K]             = Temperature
*/
double blackbodyPlanck(double temp, double freq);
__host__ __device__ void huntFloat(float* xx, int n, double x, int* jlo);
__host__ __device__ void huntDouble(double* xx, int n, double x, int* jlo);
double* remapFunction(int nOld, double* xOld, double* fOld, int nNew, double* xNew, int elow);
#endif
