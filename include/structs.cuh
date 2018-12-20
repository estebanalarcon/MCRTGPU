#ifndef STRUCTS_CUH
#define STRUCTS_CUH

#include <curand_kernel.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
typedef struct mcrtGpu{
  int numPhotons;
  int numParallelPhotons;
  int blockSize;
  float nTemp;
  float temp0;
  float temp1;
}mcrtGpu;

typedef struct Grid {
  //CoordSystem: 0 -> Cartesian
  //CoordSystem: 1 -> Spherical
  int coordSystem;
  //[0] nx
  //[1] ny
  //[2] nz
  int nCoord[3];
  double* cellWallsX;
  double* cellWallsY;
  double* cellWallsZ;
  double cellVolumes;
  int totalCells;
}Grid;

typedef struct DustOpacity{
  int iFormat;
  int numSpec;
  int* inputStyle;
  int* iQuantums;
  int* numLambdas;
  double** lambdas;
  double** kappaA;
  double** kappaS;
  double** g;
}DustOpacity;

typedef struct FrequenciesData {
  int numFrequencies;
  double* frequencies;
  double* frequenciesDnu;
}FrequenciesData;

typedef struct OpacityCoefficient {
  double alphaATotal;
  double alphaSTotal;
  double alphaTotal;
  double albedo;
  double dtau;
}OpacityCoefficient;

typedef struct Photon {
  curandState state;
  short gridPosition[3];
  short prevGridPosition[3];
  double rayPosition[3];
  double prevRayPosition[3];
  float direction[3];
  double cellWalls[3];
  double distances[3];
  bool orientations[3];
  OpacityCoefficient opacCoeff;
  unsigned short iFrequency;
  unsigned short iStar;
  bool onGrid;
  bool isScattering;
  float taupathTotal;
  float taupathGone;
  double* alphaASpec;
  double* alphaSSpec;
  float* dbCumul;
  double* enerCum;
  double* enerPart;
  double* tempLocal;
}Photon;

typedef struct TPhoton {
  curandState state;
  short gridPosition[3];
  short prevGridPosition[3];
  double rayPosition[3];
  double prevRayPosition[3];
  float direction[3];
  double cellWalls[3];
  double distances[3];
  bool orientations[3];
  OpacityCoefficient opacCoeff;
}TPhoton;

typedef struct DustDensity {
  int iFormat;
  int numSpec;
  int numCells;
  int nx;
  int ny;
  int nz;
  double**** densities;
}DustDensity;

typedef struct DustTemperature {
  int totalPositions;
  double**** cumulEner;
  double**** temperatures;
}DustTemperature;

typedef struct EmissivityDatabase {
  float nTemp;
  float temp0;
  float temp1;
  double* dbTemp;
  double** dbEnerTemp;
  double** dbLogEnerTemp;
  double*** dbEmiss;
  float*** dbCumulNorm;
}EmissivityDatabase;

typedef struct Stars {
  int numStars;
  double* radius;
  double* masses;
  double** positions;
  double** spec;
  double** specCum;
  double* luminosities;
  double* luminositiesCum;
  double luminositiesTotal;
  double* energies;
}Stars;

#endif
