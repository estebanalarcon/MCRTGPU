#include "photon.cuh"
#include "emissivity.cuh"
#include "grid.cuh"
#include "global_functions.cuh"

__device__ int signs[2] = {-1,1};

__global__ void kernelDoAbsorptionEvent(Photon* d_photons, Grid* grid, FrequenciesData* freqData, Stars* stars,
  DustDensity* dustDensity, DustOpacity* dustOpacity, EmissivityDatabase* emissivityDb, DustTemperature* dustTemperature){
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (d_photons[tid].onGrid){
      if (!d_photons[tid].isScattering){
        int ix = d_photons[tid].gridPosition[0];
        int iy = d_photons[tid].gridPosition[1];
        int iz = d_photons[tid].gridPosition[2];
        divideAbsorvedEnergy(&d_photons[tid],stars,dustDensity, dustOpacity);
        addTemperatureDecoupled(&d_photons[tid], dustDensity, grid, emissivityDb, dustTemperature);
        for (int i=0 ; i<dustDensity->numSpec ; i++){
          d_photons[tid].tempLocal[i] = dustTemperature->temperatures[i][iz][iy][ix];
        }
        d_photons[tid].iFrequency = pickRandomFreqDb(emissivityDb, &d_photons[tid], dustDensity->numSpec, freqData->numFrequencies, d_photons[tid].tempLocal, d_photons[tid].enerPart);
        getRandomDirectionSimple(&d_photons[tid], &d_photons[tid].state);
      }
      return;
    }
    return;
  }

__global__ void kernelDoScatteringEvent(Photon* d_photons){
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (d_photons[tid].onGrid){
    if (d_photons[tid].isScattering){
      getRandomDirectionSimple(&d_photons[tid], &d_photons[tid].state);
    }
    return;
  }
  return;
}

__global__ void getPhotonsOnGrid(Photon* d_photons, bool* d_photonsOnGrid){
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  d_photonsOnGrid[tid] = d_photons[tid].onGrid;
  //printf("photon%d grid=%d\n",tid,d_photonsOnGrid[tid]);
}

__host__ void transferOnGridArrayToHost(bool* d_photonsOnGrid, bool* h_photonsOnGrid, int numPhotons){
  cudaMemcpy(h_photonsOnGrid, d_photonsOnGrid,sizeof(bool)*numPhotons,cudaMemcpyDeviceToHost);
  return;
}

__host__ bool arePhotonsOnGrid(bool* d_onGrid, bool* h_onGrid, int numPhotons){
  //transferOnGridArrayToHost(d_onGrid,h_onGrid,numPhotons );
  cudaMemcpy(h_onGrid, d_onGrid,sizeof(bool)*numPhotons,cudaMemcpyDeviceToHost);
  for (int i=0 ; i<numPhotons ; i++){
    if (h_onGrid[i]){
      return true;
    }
  }
  return false;
}
/*
__global__ void convertEnergyToDecoupledTemperature(Grid* grid,Photon* photon, DustDensity* dustDensity, DustTemperature* dustTemperature, EmissivityDatabase* emissivityDb){
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  int* gridPosition = getIxyzFromIndex(grid, tid);
  int ix = photon->gridPosition[0];
  int iy = photon->gridPosition[1];
  int iz = photon->gridPosition[2];
  double ener;
  for (int i=0 ; i<dustDensity->numSpec ; i++){
    ener = dustTemperature->cumulEner[i][iz][iy][ix]/(dustDensity->densities[i][iz][iy][ix]*grid->cellVolumes);
    if (ener>0){
      dustTemperature->temperatures[i][iz][iy][ix] = computeDusttempEnergyBd(emissivityDb, ener, i);
    }else{
      dustTemperature->temperatures[i][iz][iy][ix] = 0;
    }
  }
}*/
__global__ void testId(){
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  printf("tid:%d, int %d\n",tid,5);
}
__global__ void testId2(){
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  printf("tid:%d, int %d\n",tid,2);
}

__device__ void divideAbsorvedEnergy(Photon* photon, Stars* stars,
  DustDensity* dustDensity, DustOpacity* dustOpacity){
    //printf("in divideAbsorvedEnergy\n");
    int ix = photon->gridPosition[0];
    int iy = photon->gridPosition[1];
    int iz = photon->gridPosition[2];
    double alphaA=0;
    if (dustDensity->numSpec == 1){
      photon->enerPart[0] = stars->energies[photon->iStar];
    }else{
      for (int i=0 ; i<dustDensity->numSpec ; i++){
        photon->enerPart[i] = dustDensity->densities[i][iz][iy][ix] * dustOpacity->kappaA[i][photon->iFrequency];
        alphaA += photon->enerPart[i];
      }
      for (int i=0 ; i<dustDensity->numSpec ; i++){
        photon->enerPart[i] = stars->energies[photon->iStar] * photon->enerPart[i] / alphaA;
      }
    }
  }

__device__ void addTemperatureDecoupled(Photon* photon, DustDensity* dustDensity,
  Grid* grid, EmissivityDatabase* emissivityDb, DustTemperature* dustTemperature){
    //printf("in addTemperatureDecoupled\n");

    int ix = photon->gridPosition[0];
    int iy = photon->gridPosition[1];
    int iz = photon->gridPosition[2];
    double cumen;
    for (int iSpec=0 ; iSpec < dustDensity->numSpec ; iSpec++){
      cumen = dustTemperature->cumulEner[iSpec][iz][iy][ix] / (dustDensity->densities[iSpec][iz][iy][ix]* grid->cellVolumes);
      dustTemperature->temperatures[iSpec][iz][iy][ix] = computeDusttempEnergyBd(emissivityDb, cumen, iSpec);
    }
  }

__device__ void doAbsorptionEvent(Photon* photon, Grid* grid, FrequenciesData* freqData, Stars* stars,
  DustDensity* dustDensity, DustOpacity* dustOpacity, EmissivityDatabase* emissivityDb, DustTemperature* dustTemperature){
    int ix = photon->gridPosition[0];
    int iy = photon->gridPosition[1];
    int iz = photon->gridPosition[2];
    divideAbsorvedEnergy(photon,stars,dustDensity, dustOpacity);
    addTemperatureDecoupled(photon, dustDensity, grid, emissivityDb, dustTemperature);
    for (int i=0 ; i<dustDensity->numSpec ; i++){
      photon->tempLocal[i] = dustTemperature->temperatures[i][iz][iy][ix];
    }
    photon->iFrequency = pickRandomFreqDb(emissivityDb, photon, dustDensity->numSpec, freqData->numFrequencies, photon->tempLocal, photon->enerPart);
    //photon->iFrequency = 55;
    //printf("newInu=%d\n",photon->iFrequency);
  }

__device__ double advanceToNextPositionTest(Photon* photon, Grid* grid, double cellWallsX[], double cellWallsY[], double cellWallsZ[]){
  //obtain orientations. It is 0 (left,down) or 1 (right, up)
  int ix = floor(photon->direction[0])+1.0;
  int iy = floor(photon->direction[1])+1.0;
  int iz = floor(photon->direction[2])+1.0;

  photon->orientations[0]=ix;
  photon->orientations[1]=iy;
  photon->orientations[2]=iz;

  //test shared memory
  //axis x
  photon->cellWalls[0] = cellWallsX[photon->gridPosition[0]+ix];
  //axis y
  photon->cellWalls[1] = cellWallsY[photon->gridPosition[1]+iy];
  //axis z
  photon->cellWalls[2] = cellWallsZ[photon->gridPosition[2]+iz];
  //printf("cellWallsX=%10.10lg\n",cellWallsX[photon->gridPosition[0]+ix]);


  //get 3 walls of grid position
  //getCellWalls(photon, grid, photon->gridPosition, photon->orientations);

  //distance to axis x
  photon->distances[0] = (photon->cellWalls[0] - photon->rayPosition[0]) / photon->direction[0];
  //distance to axis y
  photon->distances[1] = (photon->cellWalls[1] - photon->rayPosition[1]) / photon->direction[1];
  //distance to axis z
  photon->distances[2] = (photon->cellWalls[2] - photon->rayPosition[2]) / photon->direction[2];
  //printf("distances: %lf, %lf, %lf\n",distances[0],distances[1],distances[2]);

  //calculate min distance
  double tmp = fmin(photon->distances[0], photon->distances[1]);
  double minDistance = fmin(tmp, photon->distances[2]);
  //printf("minDistance: %lf\n", minDistance);

  //obtain minimun's axis
  //can be more than 1 (corners)
  int count = 0;
  int indexes[3] = {-1,-1,-1};
  for (int i=0 ; i<3 ; i++){
    if (photon->distances[i] == minDistance){
      indexes[count]=i;
      count++;
    }
  }
  //printf("minDistance, count: %lf, %d\n", minDistance, count);

  //update ray position
  photon->rayPosition[0] += minDistance*photon->direction[0];
  photon->rayPosition[1] += minDistance*photon->direction[1];
  photon->rayPosition[2] += minDistance*photon->direction[2];

  //avoid bug assign cellWall to ray position
  //update grid position with signs
  for (int i=0 ; i<count ; i++){
    photon->rayPosition[indexes[i]] = photon->cellWalls[indexes[i]];
    photon->gridPosition[indexes[i]] += signs[photon->orientations[indexes[i]]];
  }

  //is photon on the grid or outside?
  int nx = grid->nCoord[0];
  int ny = grid->nCoord[1];
  int nz = grid->nCoord[2];
  bool onX = (photon->gridPosition[0] >= 0) && (photon->gridPosition[0] < nx);
  bool onY = (photon->gridPosition[1] >= 0) && (photon->gridPosition[1] < ny);
  bool onZ = (photon->gridPosition[2] >= 0) && (photon->gridPosition[2] < nz);
  photon->onGrid = (onX && onY && onZ);
  return minDistance;

}

__device__ double advanceToNextPosition(Photon* photon, Grid* grid){
  //obtain orientations. It is 0 (left,down) or 1 (right, up)
  int ix = floor(photon->direction[0])+1.0;
  int iy = floor(photon->direction[1])+1.0;
  int iz = floor(photon->direction[2])+1.0;

  photon->orientations[0]=ix;
  photon->orientations[1]=iy;
  photon->orientations[2]=iz;


  //get 3 walls of grid position
  getCellWalls(photon, grid, photon->gridPosition, photon->orientations);

  //distance to axis x
  photon->distances[0] = (photon->cellWalls[0] - photon->rayPosition[0]) / photon->direction[0];
  //distance to axis y
  photon->distances[1] = (photon->cellWalls[1] - photon->rayPosition[1]) / photon->direction[1];
  //distance to axis z
  photon->distances[2] = (photon->cellWalls[2] - photon->rayPosition[2]) / photon->direction[2];
  //printf("distances: %lf, %lf, %lf\n",distances[0],distances[1],distances[2]);

  //calculate min distance
  double tmp = fmin(photon->distances[0], photon->distances[1]);
  double minDistance = fmin(tmp, photon->distances[2]);
  //printf("minDistance: %lf\n", minDistance);

  //obtain minimun's axis
  //can be more than 1 (corners)
  int count = 0;
  int indexes[3] = {-1,-1,-1};
  for (int i=0 ; i<3 ; i++){
    if (photon->distances[i] == minDistance){
      indexes[count]=i;
      count++;
    }
  }
  //printf("minDistance, count: %lf, %d\n", minDistance, count);

  //update ray position
  photon->rayPosition[0] += minDistance*photon->direction[0];
  photon->rayPosition[1] += minDistance*photon->direction[1];
  photon->rayPosition[2] += minDistance*photon->direction[2];

  //avoid bug assign cellWall to ray position
  //update grid position with signs
  for (int i=0 ; i<count ; i++){
    photon->rayPosition[indexes[i]] = photon->cellWalls[indexes[i]];
    photon->gridPosition[indexes[i]] += signs[photon->orientations[indexes[i]]];
  }

  //is photon on the grid or outside?
  photon->onGrid = photonIsOnGrid(photon, grid);
  return minDistance;
}

__device__ bool photonIsOnGrid(Photon* photon, Grid* grid){
  bool onX = (photon->gridPosition[0] >= 0) && (photon->gridPosition[0] < grid->nCoord[0]);
  bool onY = (photon->gridPosition[1] >= 0) && (photon->gridPosition[1] < grid->nCoord[1]);
  bool onZ = (photon->gridPosition[2] >= 0) && (photon->gridPosition[2] < grid->nCoord[2]);
  bool onGrid = (onX && onY && onZ);
  return onGrid;
}

__device__ void getTaupath(Photon* photon, curandState* state){
  float rn = curand_uniform(state);
  photon->taupathTotal = - log(1.0-rn);
  photon->taupathGone = 0.0;
}

__device__ void findNewFrequencyInu(Photon* photon, curandState* state, double* specCum, FrequenciesData* freqData){
  float rn = curand_uniform(state);
  int freq= (int)photon->iFrequency;
  huntDouble(specCum, freqData->numFrequencies+1, (double) rn, &freq);
  photon->iFrequency = (short)freq;
  //printf("rayInu = %d\n",rayInu);
  //return rayInu;
}

__device__ void checkUnitVector(double* direction){
    double module = sqrt(direction[0]*direction[0] + direction[1]*direction[1] + direction[2]*direction[2]);
    if (fabs(module - 1) > 1e-10){
      printf("Error unity vector\n");
    }else{
      printf("correct\n");
    }
}

__device__ void getRandomDirectionSimple(Photon* photon, curandState *state){
  //unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  float dirx, diry, dirz;
  bool equalZero = true;
  bool equalOne = true;
  float tmp, module;
  while (equalZero || equalOne){
    dirx = 2*curand_uniform(state)-1;
    diry = 2*curand_uniform(state)-1;
    dirz = 2*curand_uniform(state)-1;
    tmp = dirx*dirx + diry*diry + dirz*dirz;
    module = 1.0/sqrt(tmp);
    dirx = dirx*module;
    diry = diry*module;
    dirz = dirz*module;
    equalZero = (dirx==0.0) || (diry==0.0)|| (dirz==0.0);
    equalOne =  (dirx==1.0) || (diry==1.0)|| (dirz==1.0);
    //printf("direction %d=%lf,%lf,%lf\n",tid,photon->direction[0],photon->direction[1],photon->direction[2]);
  }

  photon->direction[0] = dirx;
  photon->direction[1] = diry;
  photon->direction[2] = dirz;

  /*printf("direction.x %lg\n",dirx);
  printf("direction.y %lg\n",diry);
  printf("direction.z %lg\n",dirz);*/
  return;
}
__device__ void findStar(Photon* photon, Stars* d_stars, curandState* state){
  photon->iStar = 0;
  if (d_stars->numStars > 1){
    float rn = curand_uniform(state);
    int istar=(int)photon->iStar;
    huntDouble(d_stars->luminositiesCum, d_stars->numStars+1, (double) rn, &istar);
    photon->iStar = (short) istar;
  }
}

__host__ void setUpPhoton(Photon* photon, int numSpec, int numFreq){

  photon->alphaASpec = (double*)malloc(sizeof(double)*numSpec);
  photon->alphaSSpec = (double*)malloc(sizeof(double)*numSpec);
  photon->dbCumul = (float*)malloc(sizeof(float)*(numFreq+1));
  photon->enerCum = (double*)malloc(sizeof(double)*(numSpec+1));
  photon->enerPart = (double*)malloc(sizeof(double)*numSpec);
  photon->tempLocal = (double*)malloc(sizeof(double)*numSpec);
  photon->onGrid = true;
  /*photon->taupathGone = 0;
  photon->taupathTotal = 0;
  photon->iFrequency = 0;
  for (int i=0 ; i<numSpec ; i++){
    photon->alphaASpec[i] = 0;
    photon->alphaSSpec[i] = 0;
  }*/
}

__host__ void freePhoton(Photon* photon){
  free(photon->alphaASpec);
  free(photon->alphaSSpec);
  free(photon->dbCumul);
  free(photon->enerCum);
  free(photon->enerPart);
  free(photon->tempLocal);
}

__device__ void calculateOpacityCoefficients(double minorDistance, Photon* photon, DustDensity* dustDensity,
  DustOpacity* dustOpacity){
    int ix = photon->prevGridPosition[0];
    int iy = photon->prevGridPosition[1];
    int iz = photon->prevGridPosition[2];
    photon->opacCoeff.alphaATotal=0;
    photon->opacCoeff.alphaSTotal=0;
    for (int iSpec=0 ; iSpec<dustDensity->numSpec ; iSpec++){
      //printf("densities: %10.10lg\n",dustDensity->densities[iSpec][iz][iy][ix]);
      //printf("kappaA: %10.10lg\n",dustOpacity->kappaA[iSpec][photon->iFrequency]);
      photon->alphaASpec[iSpec] = dustDensity->densities[iSpec][iz][iy][ix]*dustOpacity->kappaA[iSpec][photon->iFrequency];
      photon->alphaSSpec[iSpec] = dustDensity->densities[iSpec][iz][iy][ix]*dustOpacity->kappaS[iSpec][photon->iFrequency];
      photon->opacCoeff.alphaATotal += photon->alphaASpec[iSpec];
      photon->opacCoeff.alphaSTotal += photon->alphaSSpec[iSpec];
    }
    photon->opacCoeff.alphaTotal = photon->opacCoeff.alphaATotal+photon->opacCoeff.alphaSTotal;
    photon->opacCoeff.albedo = photon->opacCoeff.alphaSTotal/photon->opacCoeff.alphaTotal;
    photon->opacCoeff.dtau = photon->opacCoeff.alphaTotal * minorDistance;
    //printf("dtau: %10.10lg\n",opacCoeff->dtau);
}

__device__ void inicializePositionPhoton(Photon* photon, Stars* d_stars,
  Grid* d_grid){
  //rayPosition
  photon->rayPosition[0] = d_stars->positions[photon->iStar][0];
  photon->rayPosition[1] = d_stars->positions[photon->iStar][1];
  photon->rayPosition[2] = d_stars->positions[photon->iStar][2];

  //gridPosition
  convertRayToGrid(photon, d_grid);
}

__device__ void walkNextEvent(Photon* d_photons, Stars* d_stars, DustDensity* d_dustDensity, DustOpacity* d_dustOpacity,
    Grid* d_grid, DustTemperature* d_dustTemperature, double cellWallsX[], double cellWallsY[], double cellWallsZ[]){
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  double minorDistance, fraction, dum, addTmp;
  bool carryOn = true;
  //printf("cellWallsX 0 = %10.10lg\n",cellWallsX[0] );
  //printf("cellWallsX 1 = %10.10lg\n",cellWallsX[1] );
  while (carryOn){

    d_photons[tid].prevRayPosition[0] = d_photons[tid].rayPosition[0];
    d_photons[tid].prevRayPosition[1] = d_photons[tid].rayPosition[1];
    d_photons[tid].prevRayPosition[2] = d_photons[tid].rayPosition[2];

    d_photons[tid].prevGridPosition[0] = d_photons[tid].gridPosition[0];
    d_photons[tid].prevGridPosition[1] = d_photons[tid].gridPosition[1];
    d_photons[tid].prevGridPosition[2] = d_photons[tid].gridPosition[2];
    //printf("start advanceToNextPosition\n");
    minorDistance = advanceToNextPositionTest(&d_photons[tid], d_grid,cellWallsX,cellWallsY,cellWallsZ);
    //minorDistance = advanceToNextPosition(&d_photons[tid], d_grid);
    //if (d_photons[tid].gridPosition[0]==0 && d_photons[tid].gridPosition[1]==0 && d_photons[tid].gridPosition[2]==0){
    //  printf("GridPosition: %d, %d, %d, onGrid = %d\n",d_photons[tid].gridPosition[0],d_photons[tid].gridPosition[1],d_photons[tid].gridPosition[2], d_photons[tid].onGrid);
    //}

    calculateOpacityCoefficients(minorDistance, &d_photons[tid], d_dustDensity, d_dustOpacity);

    //printf("alphaSTotal=%10.10lg\n", opacCoeff->alphaSTotal);

    if (d_photons[tid].taupathGone + d_photons[tid].opacCoeff.dtau > d_photons[tid].taupathTotal){
      //printf("taupathGone=%lf, dtau=%lf, taupathTotal=%lf\n", d_photons[tid].taupathGone,d_photons[tid].opacCoeff.dtau,d_photons[tid].taupathTotal);
      //printf("\nReached end point!\n\n");
      fraction = (d_photons[tid].taupathTotal - d_photons[tid].taupathGone)/d_photons[tid].opacCoeff.dtau;
      //printf("fr=%lf\n", fraction)
      //update ray position

      d_photons[tid].rayPosition[0] = d_photons[tid].prevRayPosition[0] + fraction * (d_photons[tid].rayPosition[0] - d_photons[tid].prevRayPosition[0]);
      d_photons[tid].rayPosition[1] = d_photons[tid].prevRayPosition[1] + fraction * (d_photons[tid].rayPosition[1] - d_photons[tid].prevRayPosition[1]);
      d_photons[tid].rayPosition[2] = d_photons[tid].prevRayPosition[2] + fraction * (d_photons[tid].rayPosition[2] - d_photons[tid].prevRayPosition[2]);
      //update grid position
      d_photons[tid].gridPosition[0] = d_photons[tid].prevGridPosition[0];
      d_photons[tid].gridPosition[1] = d_photons[tid].prevGridPosition[1];
      d_photons[tid].gridPosition[2] = d_photons[tid].prevGridPosition[2];
      //index = getIndexFromIxyz(d_grid, photon->gridPosition);
      dum = (1.0-d_photons[tid].opacCoeff.albedo) * (d_photons[tid].taupathTotal-d_photons[tid].taupathGone) * d_stars->energies[d_photons[tid].iStar] / d_photons[tid].opacCoeff.alphaATotal;
      //printf("dum=%10.10lg\n",dum);
      for (int i=0 ; i<d_dustDensity->numSpec ; i++){
        addTmp = dum*d_photons[tid].alphaASpec[i];
        //d_dustTemperature->cumulEner[i][d_photons[tid].gridPosition[2]][d_photons[tid].gridPosition[1]][d_photons[tid].gridPosition[0]] += addTmp;

        atomicAdd(&(d_dustTemperature->cumulEner[i][d_photons[tid].gridPosition[2]][d_photons[tid].gridPosition[1]][d_photons[tid].gridPosition[0]]), addTmp);
        if (d_photons[tid].gridPosition[0]==0 && d_photons[tid].gridPosition[1]==0 && d_photons[tid].gridPosition[2]==0){
          printf("cumulEner %d, %d, %d = %10.10lg\n",d_photons[tid].gridPosition[0],d_photons[tid].gridPosition[1],d_photons[tid].gridPosition[2],
          d_dustTemperature->cumulEner[i][d_photons[tid].prevGridPosition[2]][d_photons[tid].prevGridPosition[1]][d_photons[tid].prevGridPosition[0]]);
        }
        //d_dustTemperature->cumulEner[i][d_photons[tid].gridPosition[2]][d_photons[tid].gridPosition[1]][d_photons[tid].gridPosition[0]] += addTmp;
        //printf("cumulEner=%10.10lg\n",d_dustTemperature->cumulEner[i][d_photons[tid].gridPosition[2]][d_photons[tid].gridPosition[1]][d_photons[tid].gridPosition[0]]);
      }
      carryOn = false;
    }else{
      dum = (1.0-d_photons[tid].opacCoeff.albedo) * d_photons[tid].opacCoeff.dtau * d_stars->energies[d_photons[tid].iStar] / d_photons[tid].opacCoeff.alphaATotal;
      for (int i=0 ; i<d_dustDensity->numSpec ; i++){
        addTmp = dum*d_photons[tid].alphaASpec[i];
        //d_dustTemperature->cumulEner[i][d_photons[tid].prevGridPosition[2]][d_photons[tid].prevGridPosition[1]][d_photons[tid].prevGridPosition[0]] += addTmp;
        atomicAdd(&(d_dustTemperature->cumulEner[i][d_photons[tid].prevGridPosition[2]][d_photons[tid].prevGridPosition[1]][d_photons[tid].prevGridPosition[0]]), addTmp);
        //if (d_photons[tid].prevGridPosition[0]==0 && d_photons[tid].prevGridPosition[1]==0 && d_photons[tid].prevGridPosition[2]==0){
        //  printf("cumulEner %d, %d, %d = %10.10lg\n",d_photons[tid].prevGridPosition[0],d_photons[tid].prevGridPosition[1],d_photons[tid].prevGridPosition[2],
        //  d_dustTemperature->cumulEner[i][d_photons[tid].prevGridPosition[2]][d_photons[tid].prevGridPosition[1]][d_photons[tid].prevGridPosition[0]]);
        //}
        //d_dustTemperature->cumulEner[i][d_photons[tid].prevGridPosition[2]][d_photons[tid].prevGridPosition[1]][d_photons[tid].prevGridPosition[0]] += addTmp;
        //printf("cumulEner =%10.10lg\n",d_dustTemperature->cumulEner[i][d_photons[tid].prevGridPosition[2]][d_photons[tid].prevGridPosition[1]][d_photons[tid].prevGridPosition[0]]);
      }

      d_photons[tid].taupathGone += d_photons[tid].opacCoeff.dtau;
      carryOn = d_photons[tid].onGrid;
    }
  }
  //printf("FinalRayPosition: %lf, %lf, %lf\n",d_photons[tid].rayPosition[0],d_photons[tid].rayPosition[1],d_photons[tid].rayPosition[2]);
  //printf("FinalGridPosition: %d, %d, %d\n",d_photons[tid].gridPosition[0],d_photons[tid].gridPosition[1],d_photons[tid].gridPosition[2]);
  float rn = curand_uniform(&d_photons[tid].state);
  //printf("rn=%lf\n",rn);
  d_photons[tid].isScattering = rn < d_photons[tid].opacCoeff.albedo;
  //printf("end walkEvent\n");
}
/*
__device__ void walkNextEvent(Photon* d_photons, Stars* d_stars, DustDensity* d_dustDensity, DustOpacity* d_dustOpacity,
    Grid* d_grid, DustTemperature* d_dustTemperature){
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  double minorDistance, fraction, dum, addTmp;
  bool carryOn = true;
  //printf("cellWallsX 0 = %10.10lg\n",cellWallsX[0] );
  //printf("cellWallsX 1 = %10.10lg\n",cellWallsX[1] );
  while (carryOn){

    d_photons[tid].prevRayPosition[0] = d_photons[tid].rayPosition[0];
    d_photons[tid].prevRayPosition[1] = d_photons[tid].rayPosition[1];
    d_photons[tid].prevRayPosition[2] = d_photons[tid].rayPosition[2];

    d_photons[tid].prevGridPosition[0] = d_photons[tid].gridPosition[0];
    d_photons[tid].prevGridPosition[1] = d_photons[tid].gridPosition[1];
    d_photons[tid].prevGridPosition[2] = d_photons[tid].gridPosition[2];
    //printf("start advanceToNextPosition\n");
    //minorDistance = advanceToNextPositionTest(&d_photons[tid], d_grid,cellWallsX,cellWallsY,cellWallsZ);
    minorDistance = advanceToNextPosition(&d_photons[tid], d_grid);
    if (d_photons[tid].gridPosition[0]==0 && d_photons[tid].gridPosition[1]==0 && d_photons[tid].gridPosition[2]==0){
      printf("GridPosition: %d, %d, %d, onGrid = %d\n",d_photons[tid].gridPosition[0],d_photons[tid].gridPosition[1],d_photons[tid].gridPosition[2], d_photons[tid].onGrid);
    }

    if (d_photons[tid].onGrid){
      calculateOpacityCoefficients(minorDistance, &d_photons[tid], d_dustDensity, d_dustOpacity);

      //printf("alphaSTotal=%10.10lg\n", opacCoeff->alphaSTotal);

      if (d_photons[tid].taupathGone + d_photons[tid].opacCoeff.dtau > d_photons[tid].taupathTotal){
        //printf("taupathGone=%lf, dtau=%lf, taupathTotal=%lf\n", d_photons[tid].taupathGone,d_photons[tid].opacCoeff.dtau,d_photons[tid].taupathTotal);
        //printf("\nReached end point!\n\n");
        fraction = (d_photons[tid].taupathTotal - d_photons[tid].taupathGone)/d_photons[tid].opacCoeff.dtau;
        //printf("fr=%lf\n", fraction)
        //update ray position

        d_photons[tid].rayPosition[0] = d_photons[tid].prevRayPosition[0] + fraction * (d_photons[tid].rayPosition[0] - d_photons[tid].prevRayPosition[0]);
        d_photons[tid].rayPosition[1] = d_photons[tid].prevRayPosition[1] + fraction * (d_photons[tid].rayPosition[1] - d_photons[tid].prevRayPosition[1]);
        d_photons[tid].rayPosition[2] = d_photons[tid].prevRayPosition[2] + fraction * (d_photons[tid].rayPosition[2] - d_photons[tid].prevRayPosition[2]);
        //update grid position
        d_photons[tid].gridPosition[0] = d_photons[tid].prevGridPosition[0];
        d_photons[tid].gridPosition[1] = d_photons[tid].prevGridPosition[1];
        d_photons[tid].gridPosition[2] = d_photons[tid].prevGridPosition[2];
        //index = getIndexFromIxyz(d_grid, photon->gridPosition);
        dum = (1.0-d_photons[tid].opacCoeff.albedo) * (d_photons[tid].taupathTotal-d_photons[tid].taupathGone) * d_stars->energies[d_photons[tid].iStar] / d_photons[tid].opacCoeff.alphaATotal;
        //printf("dum=%10.10lg\n",dum);
        for (int i=0 ; i<d_dustDensity->numSpec ; i++){
          addTmp = dum*d_photons[tid].alphaASpec[i];
          //d_dustTemperature->cumulEner[i][d_photons[tid].gridPosition[2]][d_photons[tid].gridPosition[1]][d_photons[tid].gridPosition[0]] += addTmp;

          atomicAdd(&(d_dustTemperature->cumulEner[i][d_photons[tid].gridPosition[2]][d_photons[tid].gridPosition[1]][d_photons[tid].gridPosition[0]]), addTmp);
          if (d_photons[tid].gridPosition[0]==0 && d_photons[tid].gridPosition[1]==0 && d_photons[tid].gridPosition[2]==0){
            printf("cumulEner %d, %d, %d = %10.10lg\n",d_photons[tid].gridPosition[0],d_photons[tid].gridPosition[1],d_photons[tid].gridPosition[2],
            d_dustTemperature->cumulEner[i][d_photons[tid].prevGridPosition[2]][d_photons[tid].prevGridPosition[1]][d_photons[tid].prevGridPosition[0]]);
          }
          //d_dustTemperature->cumulEner[i][d_photons[tid].gridPosition[2]][d_photons[tid].gridPosition[1]][d_photons[tid].gridPosition[0]] += addTmp;
          //printf("cumulEner=%10.10lg\n",d_dustTemperature->cumulEner[i][d_photons[tid].gridPosition[2]][d_photons[tid].gridPosition[1]][d_photons[tid].gridPosition[0]]);
        }
        carryOn = false;
      }else{
        dum = (1.0-d_photons[tid].opacCoeff.albedo) * d_photons[tid].opacCoeff.dtau * d_stars->energies[d_photons[tid].iStar] / d_photons[tid].opacCoeff.alphaATotal;
        for (int i=0 ; i<d_dustDensity->numSpec ; i++){
          addTmp = dum*d_photons[tid].alphaASpec[i];
          //d_dustTemperature->cumulEner[i][d_photons[tid].prevGridPosition[2]][d_photons[tid].prevGridPosition[1]][d_photons[tid].prevGridPosition[0]] += addTmp;
          atomicAdd(&(d_dustTemperature->cumulEner[i][d_photons[tid].prevGridPosition[2]][d_photons[tid].prevGridPosition[1]][d_photons[tid].prevGridPosition[0]]), addTmp);
          if (d_photons[tid].prevGridPosition[0]==0 && d_photons[tid].prevGridPosition[1]==0 && d_photons[tid].prevGridPosition[2]==0){
            printf("cumulEner %d, %d, %d = %10.10lg\n",d_photons[tid].prevGridPosition[0],d_photons[tid].prevGridPosition[1],d_photons[tid].prevGridPosition[2],
            d_dustTemperature->cumulEner[i][d_photons[tid].prevGridPosition[2]][d_photons[tid].prevGridPosition[1]][d_photons[tid].prevGridPosition[0]]);
          }
          //d_dustTemperature->cumulEner[i][d_photons[tid].prevGridPosition[2]][d_photons[tid].prevGridPosition[1]][d_photons[tid].prevGridPosition[0]] += addTmp;
          //printf("cumulEner =%10.10lg\n",d_dustTemperature->cumulEner[i][d_photons[tid].prevGridPosition[2]][d_photons[tid].prevGridPosition[1]][d_photons[tid].prevGridPosition[0]]);
        }

        d_photons[tid].taupathGone += d_photons[tid].opacCoeff.dtau;
      }
      //if photon is outside of grid
    }else{
      carryOn = false;
    }
          //printf("Ongrid? = %d\n",d_photons[tid].onGrid);
  }
  //printf("FinalRayPosition: %lf, %lf, %lf\n",d_photons[tid].rayPosition[0],d_photons[tid].rayPosition[1],d_photons[tid].rayPosition[2]);
  //printf("FinalGridPosition: %d, %d, %d\n",d_photons[tid].gridPosition[0],d_photons[tid].gridPosition[1],d_photons[tid].gridPosition[2]);
  float rn = curand_uniform(&d_photons[tid].state);
  //printf("rn=%lf\n",rn);
  d_photons[tid].isScattering = rn < d_photons[tid].opacCoeff.albedo;
  //printf("end walkEvent\n");
}*/

__global__ void kernelWalkNextEvent(Photon* d_photons, Stars* d_stars, DustDensity* d_dustDensity, DustOpacity* d_dustOpacity,
  Grid* d_grid, DustTemperature* d_dustTemperature){
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (d_photons[tid].onGrid){
      //printf("startGridPosition %d: %d, %d, %d, ongrid=%d\n",tid,d_photons[tid].gridPosition[0],d_photons[tid].gridPosition[1],d_photons[tid].gridPosition[2],d_photons[tid].onGrid);

      double minorDistance, fraction, dum, addTmp;
      bool carryOn = true;

      while (carryOn){
        //printf("\n");
        d_photons[tid].prevRayPosition[0] = d_photons[tid].rayPosition[0];
        d_photons[tid].prevRayPosition[1] = d_photons[tid].rayPosition[1];
        d_photons[tid].prevRayPosition[2] = d_photons[tid].rayPosition[2];

        d_photons[tid].prevGridPosition[0] = d_photons[tid].gridPosition[0];
        d_photons[tid].prevGridPosition[1] = d_photons[tid].gridPosition[1];
        d_photons[tid].prevGridPosition[2] = d_photons[tid].gridPosition[2];

        minorDistance = advanceToNextPosition(&d_photons[tid], d_grid);
        if (d_photons[tid].onGrid){
          calculateOpacityCoefficients(minorDistance, &d_photons[tid], d_dustDensity, d_dustOpacity);

          //printf("alphaSTotal=%10.10lg\n", opacCoeff->alphaSTotal);

          if (d_photons[tid].taupathGone + d_photons[tid].opacCoeff.dtau > d_photons[tid].taupathTotal){
            //printf("taupathGone=%lf, dtau=%lf, taupathTotal=%lf\n", d_photons[tid].taupathGone,d_photons[tid].opacCoeff.dtau,d_photons[tid].taupathTotal);
            //printf("\nReached end point!\n\n");
            fraction = (d_photons[tid].taupathTotal - d_photons[tid].taupathGone)/d_photons[tid].opacCoeff.dtau;
            //printf("fr=%lf\n", fraction)
            //update ray position

            d_photons[tid].rayPosition[0] = d_photons[tid].prevRayPosition[0] + fraction * (d_photons[tid].rayPosition[0] - d_photons[tid].prevRayPosition[0]);
            d_photons[tid].rayPosition[1] = d_photons[tid].prevRayPosition[1] + fraction * (d_photons[tid].rayPosition[1] - d_photons[tid].prevRayPosition[1]);
            d_photons[tid].rayPosition[2] = d_photons[tid].prevRayPosition[2] + fraction * (d_photons[tid].rayPosition[2] - d_photons[tid].prevRayPosition[2]);
            //update grid position
            d_photons[tid].gridPosition[0] = d_photons[tid].prevGridPosition[0];
            d_photons[tid].gridPosition[1] = d_photons[tid].prevGridPosition[1];
            d_photons[tid].gridPosition[2] = d_photons[tid].prevGridPosition[2];
            //index = getIndexFromIxyz(d_grid, photon->gridPosition);
            dum = (1-d_photons[tid].opacCoeff.albedo) * (d_photons[tid].taupathTotal-d_photons[tid].taupathGone) * d_stars->energies[d_photons[tid].iStar] / d_photons[tid].opacCoeff.alphaATotal;
            //printf("dum=%10.10lg\n",dum);
            for (int i=0 ; i<d_dustDensity->numSpec ; i++){
              addTmp = dum*d_photons[tid].alphaASpec[i];
              doubleAtomicAdd(&(d_dustTemperature->cumulEner[i][d_photons[tid].gridPosition[2]][d_photons[tid].gridPosition[1]][d_photons[tid].gridPosition[0]]), addTmp);
              //printf("cumulEner=%10.10lg\n",d_dustTemperature->cumulEner[i][d_photons[tid].gridPosition[2]][d_photons[tid].gridPosition[1]][d_photons[tid].gridPosition[0]]);
            }
            carryOn = false;
          }else{
            //index = getIndexFromIxyz(d_grid, photon->prevGridPosition);
            //printf("dum=%10.10lg\n",dum);
            dum = (1-d_photons[tid].opacCoeff.albedo) * d_photons[tid].opacCoeff.dtau * d_stars->energies[d_photons[tid].iStar] / d_photons[tid].opacCoeff.alphaATotal;
            for (int i=0 ; i<d_dustDensity->numSpec ; i++){
              addTmp = dum*d_photons[tid].alphaASpec[i];
              doubleAtomicAdd(&(d_dustTemperature->cumulEner[i][d_photons[tid].prevGridPosition[2]][d_photons[tid].prevGridPosition[1]][d_photons[tid].prevGridPosition[0]]), addTmp);
              //printf("cumulEner =%10.10lg\n",d_dustTemperature->cumulEner[i][d_photons[tid].prevGridPosition[2]][d_photons[tid].prevGridPosition[1]][d_photons[tid].prevGridPosition[0]]);
            }
            d_photons[tid].taupathGone += d_photons[tid].opacCoeff.dtau;
          }
          //if photon is outside of grid
        }else{
          carryOn = false;
        }
              //printf("Ongrid? = %d\n",d_photons[tid].onGrid);
      }
      //printf("FinalGridPosition %d: %d, %d, %d, ongrid=%d\n",tid,d_photons[tid].gridPosition[0],d_photons[tid].gridPosition[1],d_photons[tid].gridPosition[2],d_photons[tid].onGrid);
      //printf("FinalRayPosition %d: %lf, %lf, %lf\n",tid, d_photons[tid].rayPosition[0],d_photons[tid].rayPosition[1],d_photons[tid].rayPosition[2]);
      float rn = curand_uniform(&d_photons[tid].state);
      //printf("rn=%lf\n",rn);
      d_photons[tid].isScattering = rn < d_photons[tid].opacCoeff.albedo;
    }
}

__global__ void inicializeInicialPhoton(Photon* d_photons, FrequenciesData* d_freqData, Stars* d_stars, Grid* d_grid,
  DustDensity* d_dustDensity, DustOpacity* d_dustOpacity){
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    //curandState state;
    curand_init((unsigned long long)clock()*tid, tid, 0, &d_photons[tid].state);
    //photons[tid] = setUpPhoton(d_dustDensity->numSpec, d_freqData->numFrequencies);
    findStar(&d_photons[tid],d_stars,&d_photons[tid].state);
    inicializePositionPhoton(&d_photons[tid], d_stars, d_grid);
    getRandomDirectionSimple(&d_photons[tid], &d_photons[tid].state);
    findNewFrequencyInu(&d_photons[tid],&d_photons[tid].state, d_stars->specCum[d_photons[tid].iStar], d_freqData);
    getTaupath(&d_photons[tid], &d_photons[tid].state);
    d_photons[tid].onGrid = true;
    //printf("iFrequency%d, iStar%d, taupath%lf, direction:%lf,%lf,%lf\n",
    //d_photons[tid].iFrequency, d_photons[tid].iStar, d_photons[tid].taupathTotal,
    //d_photons[tid].direction[0],d_photons[tid].direction[1],d_photons[tid].direction[2]);
}
/*
__global__ void restartValuesPhoton(Photon* d_photons){

}*/


__host__ Photon* allocatePhotons(int numPhotons, int numSpec, int numFreq){
  Photon* photons = (Photon*)malloc(sizeof(Photon)*numPhotons);
  for (int i=0 ; i< numPhotons ; i++){
    setUpPhoton(&photons[i], numSpec, numFreq);
  }
  return photons;
}

__host__ void deallocatePhotons(Photon* photons, int numPhotons){
  for (int i=0 ; i< numPhotons ; i++){
    freePhoton(&photons[i]);
  }
  free(photons);
}

__host__ Photon* photonsTransferToDevice(Photon* h_photons, int numPhotons, int numSpec, int numFreq){
  printf("Transfer photons to device...\n");
  Photon* d_photons;
  cudaMalloc((void**)&(d_photons), sizeof(Photon)*numPhotons );
  Photon* photons = (Photon*)malloc(sizeof(Photon)*numPhotons);
  for (int i=0 ; i<numPhotons ; i++){
    cudaMalloc((void**) &photons[i].alphaASpec, sizeof(double)*numSpec);
    cudaMalloc((void**) &photons[i].alphaSSpec, sizeof(double)*numSpec);
    cudaMalloc((void**) &photons[i].enerPart, sizeof(double)*numSpec);
    cudaMalloc((void**) &photons[i].tempLocal, sizeof(double)*numSpec);
    cudaMalloc((void**) &photons[i].enerCum, sizeof(double)*(numSpec+1));
    cudaMalloc((void**) &photons[i].dbCumul, sizeof(float)*(numFreq+1));
  }
  cudaMemcpy(d_photons, photons,sizeof(Photon)*numPhotons,cudaMemcpyHostToDevice);
  free(photons);
  return d_photons;
}
/*
__global__ void printfPhotons(Photon* d_photons){
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  d_photons[tid].opacCoeff.dtau = tid;
  printf("d_photons[tid].opacCoeff.dtau %lf\n",d_photons[tid].opacCoeff.dtau);
  //printf("Hola\n");
}*/

__global__ void launchPhotons(Photon* d_photons, FrequenciesData* d_freqData, Stars* d_stars, Grid* d_grid,
  DustDensity* d_dustDensity, DustOpacity* d_dustOpacity,
  EmissivityDatabase* d_emissivityDb, DustTemperature* d_dustTemperature){
  __shared__ double cellWallsX[257];
  __shared__ double cellWallsY[257];
  __shared__ double cellWallsZ[257];

  int scatteringMode = 1;
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

  if (threadIdx.x < 63){
    for (int i=0 ; i<4 ; i++){
      int index = threadIdx.x*4+i;
      cellWallsX[index] = d_grid->cellWallsX[index];
      cellWallsY[index] = d_grid->cellWallsY[index];
      cellWallsZ[index] = d_grid->cellWallsZ[index];
      //printf("cellWallsX %d = %10.10lg\n",index,cellWallsX[index] );
    }
  }
  if (threadIdx.x == 63){
    for (int i=0 ; i<6 ; i++){
      int index = threadIdx.x*4+i;
      cellWallsX[index] = d_grid->cellWallsX[index];
      cellWallsY[index] = d_grid->cellWallsY[index];
      cellWallsZ[index] = d_grid->cellWallsZ[index];
    }
  }
  __syncthreads();
  //printf("hola\n");
  walkNextEvent(d_photons, d_stars, d_dustDensity, d_dustOpacity, d_grid, d_dustTemperature, cellWallsX, cellWallsY, cellWallsZ);
  //walkNextEvent(d_photons, d_stars, d_dustDensity, d_dustOpacity, d_grid, d_dustTemperature);
  while(d_photons[tid].onGrid){

    if (d_photons[tid].isScattering){
      //Do Scattering event
      if (scatteringMode == 1){
        //random direction
        //printf("scaterring\n");
        getRandomDirectionSimple(&d_photons[tid], &d_photons[tid].state);
      }else if (scatteringMode == 2){
        //do henyey_greenstein_direction
      }

    }else{
      //printf("hola2\n");

      doAbsorptionEvent(&d_photons[tid], d_grid, d_freqData, d_stars, d_dustDensity,d_dustOpacity, d_emissivityDb, d_dustTemperature);
      //printf("newFreq = %d\n",d_photons[tid].iFrequency);
      //printf("start getRandomDirectionSimple\n");
      getRandomDirectionSimple(&d_photons[tid], &d_photons[tid].state);
      //printf("end getRandomDirectionSimple\n");
    }
    //printf("start getTaupath\n");
    getTaupath(&d_photons[tid], &d_photons[tid].state);
    //printf("start walkNextEvent, onGrid=%d\n",d_photons[tid].onGrid);
    walkNextEvent(d_photons, d_stars, d_dustDensity, d_dustOpacity, d_grid, d_dustTemperature,cellWallsX, cellWallsY, cellWallsZ);
    //walkNextEvent(d_photons, d_stars, d_dustDensity, d_dustOpacity, d_grid, d_dustTemperature);
    //printf("onGrid? %d\n",d_photons[tid].onGrid);
  }
}
