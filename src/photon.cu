#include "photon.cuh"
#include "emissivity.cuh"
#include "grid.cuh"
#include "global_functions.cuh"
#include "constants.cuh"

__device__ int signs[2] = {-1,1};
__device__ float precision[3] = {0,0.000001,-0.000001};
__device__ int valuesOrientations[3] = {0,1,1};

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
  //int ix = floor(photon->direction[0])+1.0;
  //int iy = floor(photon->direction[1])+1.0;
  //int iz = floor(photon->direction[2])+1.0;

  //photon->orientations[0]=ix;
  //photon->orientations[1]=iy;
  //photon->orientations[2]=iz;


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

__device__ void checkUnitVector(float x, float y, float z){
    float module = sqrt(x*x + y*y + z*z);
    if (fabs(module - 1.0) > 1e-6){
        printf("%lf %lf %lf Error unity vector\n",x,y,z);
    }else{
      //printf("%2.8lg %2.8lg %2.8lg correct\n",x,y,z);
    }
}

__device__ void getRandomDirectionSimple(Photon* photon, curandState *state){
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  float dirx=0, diry=0, dirz=0;
  float rnX;
  float rnY;
  float rnZ;
  //ol correctRandoms = false;
  bool equalZero = true;
  //bool equalOne = true;
  float tmp, module;


  while (equalZero){
    tmp = 2.0;
    while(tmp > 1){
      rnX=curand_uniform(&photon->state);
      rnY=curand_uniform(&photon->state);
      rnZ=curand_uniform(&photon->state);
      //correctRandoms = (rnX != 0.5) && (rnY != 0.5) && (rnZ != 0.5);

      dirx = 2*rnX-1;
      diry = 2*rnY-1;
      dirz = 2*rnZ-1;

      tmp = dirx*dirx + diry*diry + dirz*dirz;
      if (tmp < 0.0001){
        tmp = 2.0;
      }
    }
    //printf("direction =%lf,%lf,%lf\n",dirx,diry,dirz);
    //printf("direction =%lf,%lf,%lf\n",dirx,diry,dirz);
    equalZero = (dirx==0.0) || (diry==0.0)|| (dirz==0.0);
  }
    //printf("tmp %d= %f\n",tid,tmp);

  module = 1.0/sqrtf(tmp);
  dirx = dirx*module;
  diry = diry*module;
  dirz = dirz*module;

  //equalOne =  (dirx==1.0) || (diry==1.0)|| (dirz==1.0);
  //printf("direction %d=%lf,%lf,%lf\n",tid,photon->direction[0],photon->direction[1],photon->direction[2]);

  checkUnitVector(dirx,diry,dirz);


  //get orientations
  //obtain orientations. It is 0 (left,down) or 1 (right, up)
  int ix = floor(dirx)+1.0;
  int iy = floor(diry)+1.0;
  int iz = floor(dirz)+1.0;

  photon->orientations[0]=valuesOrientations[ix];
  photon->orientations[1]=valuesOrientations[iy];
  photon->orientations[2]=valuesOrientations[iz];

  photon->direction[0] = dirx;
  photon->direction[1] = diry;
  photon->direction[2] = dirz;

  //printf("direction=%lf,%lf,%lf\norientations %d, %d, %d\n",dirx,diry,dirz,photon->orientations[0],photon->orientations[1],photon->orientations[2]);
  return;
}

__device__ void radmcHenveyGreensteinDirection(Photon* photon, DustOpacity* dustOpacity, int iSpec){
  float newx,newy,newz;
  float temp, g2, sign;
  bool equalZero = true;
  //printf("dustOpacity->g[iSpec][photon->iFrequency]=%10.10lg\n",dustOpacity->g[iSpec][photon->iFrequency]);
  double g = dustOpacity->g[iSpec][photon->iFrequency];

  //get random numbers != 0.5 , and 2*random-1 != 0
  float rnX = 0.5;
  float rnY = 0.5;
  float rnZ = 0.5;

  while (rnX != 0.5){
    rnX=curand_uniform(&photon->state);
  }

  newx = 2*rnX-1;
  if (g > 0){
    //float rn = curand_uniform(&photon->state);
    g2 = g*g;
    temp = (1.0-g2)/(1.0 + g*newx);
    newx = (1.0+g2 - temp*temp)/(2.0*g);
    newx = fmaxf(newx,-1.0);
    newx = fminf(newx,1.0);
  }


  float l2 = 2.0;
  while(l2>1.0){
      rnY=curand_uniform(&photon->state);
      rnZ=curand_uniform(&photon->state);
    newy =2*rnY-1;
    newz =2*rnZ-1;
    l2 = newy*newy + newz*newz;
    l2 = (l2 < 0.0001 ? 2.0 : l2);
  }

  float linv = sqrtf((1.0-newx*newx)/l2);
  newy= newy*linv;
  newz= newz*linv;

  float oldx=photon->direction[0];
  float oldy=photon->direction[1];
  float oldz=photon->direction[2];

  //rotateVector
  float l = sqrtf(oldx*oldx+oldy*oldy);
  float vx = l*newx-oldz*newz;
  float vy=newy;
  float vz=oldz*newx+l*newz;
  newx = vx;
  newz = vz;
  if (l>1e-6){
    float dx=oldx/l;
    float dy=oldy/l;
    newx=dx*vx-dy*vy;
    newy=dy*vx+dx*vy;
  }
  checkUnitVector(newx,newy,newz);
  //get orientations
  //obtain orientations. It is 0 (left,down) or 1 (right, up)
  int ix = floorf(newx)+1.0;
  int iy = floorf(newy)+1.0;
  int iz = floorf(newz)+1.0;

  photon->orientations[0]=valuesOrientations[ix];
  photon->orientations[1]=valuesOrientations[iy];
  photon->orientations[2]=valuesOrientations[iz];
  //printf("floor: %d, %d, %d\n",ix,iy,iz);
  /*if (ix==2 || iy==2 ||iz==2 ){
    printf("actual: dirx=%lf diry=%lf dirz=%lf\nnew: dirx=%lf diry=%lf dirz=%lf\n",photon->direction[0],photon->direction[1],photon->direction[2],newx,newy,newz);
  }

  if (ix==2 || iy==2 ||iz==2 ){
    checkUnitVector(newx,newy,newz);
    //printf("newx=%2.8lg newy=%2.8lg newz=%2.8lg\n",newx,newy,newz);
  }*/
  photon->direction[0]=newx;
  photon->direction[1]=newy;
  photon->direction[2]=newz;
  return;

}

__device__ void getHenveyGreensteinDirection(Photon* photon, DustOpacity* dustOpacity, int iSpec){
  //printf("iSpec=%d, ifreq=%d\n",iSpec,photon->iFrequency);
  float newx,newy,newz;
  float temp, g2, sign;
  //printf("dustOpacity->g[iSpec][photon->iFrequency]=%10.10lg\n",dustOpacity->g[iSpec][photon->iFrequency]);
  double g = dustOpacity->g[iSpec][photon->iFrequency];
  float cosTheta = 2*curand_uniform(&photon->state)-1;
  if (g > 0){
    //float rn = curand_uniform(&photon->state);
    g2 = g*g;
    temp = (1.0-g2)/(1.0 + g*cosTheta);
    cosTheta = (1.0+g2 - temp*temp)/(2.0*g);
    cosTheta = fmaxf(cosTheta,-1.0);
    cosTheta = fminf(cosTheta,1.0);
  }


  float sinTheta = sqrtf(1-cosTheta*cosTheta);
  float psi = 2*PI*curand_uniform(&photon->state);
  float cosPsi = cosf(psi);
  float sinPsi = sqrtf(1-cosPsi*cosPsi);
  float sinTcosP = sinTheta*cosPsi;
  float sinTsinP = sinTheta*sinPsi;
  float last_ux = photon->direction[0];
  float last_uy = photon->direction[1];
  float last_uz = photon->direction[2];


  if (fabsf(last_uz) >= 0.99999  || fabsf(last_uy) >= 0.99999 || fabsf(last_ux) >= 0.99999){  //|| fabsf(last_uy) >= 0.99999 || fabsf(last_ux) >= 0.99999
    sign = ((last_uz) >= 0 ? 1 : -1);
    newx = sinTcosP;
    newy = sinTsinP;
    newz = cosTheta*sign;
  }else{
    temp = 1.0 / sqrtf(1 - last_uz * last_uz);
    newx = (sinTcosP * last_ux * last_uz - sinTsinP * last_uy) * temp + last_ux * cosTheta;
    newy = (sinTcosP * last_uy * last_uz + sinTsinP * last_ux) * temp + last_uy * cosTheta;
    newz = (-sinTcosP / temp) + last_uz * cosTheta;
  }
  checkUnitVector(newx,newy,newz);
  /*if (ix==2 || iy==2 ||iz==2 ){
    printf("actual: dirx=%lf diry=%lf dirz=%lf\nnew: dirx=%lf diry=%lf dirz=%lf\n",photon->direction[0],photon->direction[1],photon->direction[2],newx,newy,newz);
  }*/
  /*if (ix==2 || iy==2 ||iz==2 ){
    checkUnitVector(newx,newy,newz);
    //printf("newx=%2.8lg newy=%2.8lg newz=%2.8lg\n",newx,newy,newz);
  }*/
  photon->direction[0]=newx;
  photon->direction[1]=newy;
  photon->direction[2]=newz;
  return;
}
/*
__device__ void getHenveyGreensteinDirection(Photon* photon, DustOpacity* dustOpacity, double g){
  float newx,newy,newz;
  bool equalZero = true;
  bool equalOne = true;
  float rnx = curand_uniform(&photon->state);
  if (g != 0){
    float g2 = g*g;
    float tmp = (1.0-g2)/(1-g+2*g*rn);
    newx = (0.5*g)*(1.0+g2 - pow(tmp,2));
  }else{
    newx = 2*curand_uniform(&photon->state)-1;
  }

}*/

__device__ int findSpeciesToScattering(Photon* photon, int numSpec){
  int iSpec=0;
  photon->alphaCum[0] = 0.0;
  for (int i=0 ; i<numSpec ; i++){
    photon->alphaCum[i+1] = photon->alphaCum[i]+photon->alphaSSpec[i];
  }
  for (int i=0 ; i<numSpec ; i++){
    photon->alphaCum[i] = photon->alphaCum[i]+photon->alphaCum[numSpec];
  }
  photon->alphaCum[numSpec] = 1.0;
  float rn = curand_uniform(&photon->state);
  huntDouble(photon->alphaCum, numSpec+1, (double) rn, &iSpec);
  return iSpec;
}

__device__ void findStar(Photon* photon, Stars* d_stars, curandState* state){
  photon->iStar = 0;
  if (d_stars->numStars > 1){
    float rn = curand_uniform(state);
    int istar=(int)photon->iStar;
    //printf("star lumcum %10.10lg %10.10lg %10.10lg\n",d_stars->luminositiesCum[0],d_stars->luminositiesCum[1],d_stars->luminositiesCum[2]);
    //for (int i=0 ; i<d_stars->numStars+1 ; i++){
    //  printf("%10.10lg ",d_stars->luminositiesCum[i]);
    //}
  //  printf("\n");
    huntDouble(d_stars->luminositiesCum, d_stars->numStars+1, (double) rn, &istar);
    photon->iStar = (short) istar;
  }
}

__host__ void setUpPhoton(Photon* photon, int numSpec, int numFreq){
  photon->alphaCum = (double*)malloc(sizeof(double)*(numSpec+1));
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
      //printf("densities iSpec%d: %10.10lg\n",iSpec,dustDensity->densities[iSpec][iz][iy][ix]);
      //printf("kappaA iSpec%d: %10.10lg\n",iSpec,dustOpacity->kappaA[iSpec][photon->iFrequency]);
      photon->alphaASpec[iSpec] = dustDensity->densities[iSpec][iz][iy][ix]*dustOpacity->kappaA[iSpec][photon->iFrequency];
      photon->alphaSSpec[iSpec] = dustDensity->densities[iSpec][iz][iy][ix]*dustOpacity->kappaS[iSpec][photon->iFrequency];
      photon->opacCoeff.alphaATotal += photon->alphaASpec[iSpec];
      photon->opacCoeff.alphaSTotal += photon->alphaSSpec[iSpec];
      //printf("iSpec%d alphaATotal: %10.10lg\n",iSpec, photon->opacCoeff.alphaATotal);
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
    Grid* d_grid, DustTemperature* d_dustTemperature){ //double cellWallsX[], double cellWallsY[], double cellWallsZ[]
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  double minorDistance, fraction, dum, addTmp;
  bool carryOn = true;
  //printf("in walkEvent\n");
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
    //printf("newPosition: %d, %d, %d\n",d_photons[tid].gridPosition[0],d_photons[tid].gridPosition[1],d_photons[tid].gridPosition[2]);
    //if (d_photons[tid].gridPosition[0]==0 && d_photons[tid].gridPosition[1]==0 && d_photons[tid].gridPosition[2]==0){
    //  printf("GridPosition: %d, %d, %d, onGrid = %d\n",d_photons[tid].gridPosition[0],d_photons[tid].gridPosition[1],d_photons[tid].gridPosition[2], d_photons[tid].onGrid);
    //}

    calculateOpacityCoefficients(minorDistance, &d_photons[tid], d_dustDensity, d_dustOpacity);
    //printf("alphaATotal=%10.10lg\n", d_photons[tid].opacCoeff.alphaATotal);
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
        //printf("numSpec=%d\n",d_dustDensity->numSpec);
        addTmp = dum*d_photons[tid].alphaASpec[i];
        //d_dustTemperature->cumulEner[i][d_photons[tid].gridPosition[2]][d_photons[tid].gridPosition[1]][d_photons[tid].gridPosition[0]] += addTmp;

        atomicAdd(&(d_dustTemperature->cumulEner[i][d_photons[tid].gridPosition[2]][d_photons[tid].gridPosition[1]][d_photons[tid].gridPosition[0]]), addTmp);
        /*if (d_photons[tid].gridPosition[0]==0 && d_photons[tid].gridPosition[1]==0 && d_photons[tid].gridPosition[2]==0){
          printf("cumulEner %d, %d, %d = %10.10lg\n",d_photons[tid].gridPosition[0],d_photons[tid].gridPosition[1],d_photons[tid].gridPosition[2],
          d_dustTemperature->cumulEner[i][d_photons[tid].prevGridPosition[2]][d_photons[tid].prevGridPosition[1]][d_photons[tid].prevGridPosition[0]]);
        }*/
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
              printf("numSpec=%d\n",d_dustDensity->numSpec);
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
    //printf("iStar=%d\n",d_photons[tid].iStar);
    inicializePositionPhoton(&d_photons[tid], d_stars, d_grid);
    //printf("GridPosition %d: %d, %d, %d\n",tid,d_photons[tid].gridPosition[0],d_photons[tid].gridPosition[1],d_photons[tid].gridPosition[2],d_photons[tid].onGrid);
    //printf("RayPosition %d: %lf, %lf, %lf\n",tid, d_photons[tid].rayPosition[0],d_photons[tid].rayPosition[1],d_photons[tid].rayPosition[2]);
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
    cudaMalloc((void**) &photons[i].alphaCum, sizeof(double)*(numSpec+1));
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
  EmissivityDatabase* d_emissivityDb, DustTemperature* d_dustTemperature, SimulationParameters* d_params){
  //__shared__ double cellWallsX[257];
  //__shared__ double cellWallsY[257];
  //__shared__ double cellWallsZ[257];
  //printf("scatteringMode = %d\n",d_params->scatteringMode);
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
/*
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
  __syncthreads();*/
  //printf("hola\n");
  walkNextEvent(d_photons, d_stars, d_dustDensity, d_dustOpacity, d_grid, d_dustTemperature); // cellWallsX, cellWallsY, cellWallsZ
  //walkNextEvent(d_photons, d_stars, d_dustDensity, d_dustOpacity, d_grid, d_dustTemperature);
  while(d_photons[tid].onGrid){

    if (d_photons[tid].isScattering){
      //Do Scattering event
      if (d_params->scatteringMode == 1){
        //random direction
        //printf("scaterring\n");
        //printf("isotropic\n");
        getRandomDirectionSimple(&d_photons[tid], &d_photons[tid].state);
      }else if (d_params->scatteringMode == 2){
        //printf("anisotropic\n");

        //do henyey_greenstein_direction
        int iSpec = findSpeciesToScattering(&d_photons[tid], d_dustDensity->numSpec);
        //printf("dustOpacity->g[iSpec][photon->iFrequency] %10.10lg\n",d_dustOpacity->g[0][10]);
        radmcHenveyGreensteinDirection(&d_photons[tid], d_dustOpacity, iSpec);
        //getHenveyGreensteinDirection(&d_photons[tid], d_dustOpacity, iSpec);
      }

    }else{
      //printf("absoprtion\n");

      doAbsorptionEvent(&d_photons[tid], d_grid, d_freqData, d_stars, d_dustDensity,d_dustOpacity, d_emissivityDb, d_dustTemperature);
      //printf("newFreq = %d\n",d_photons[tid].iFrequency);
      //printf("start getRandomDirectionSimple\n");
      getRandomDirectionSimple(&d_photons[tid], &d_photons[tid].state);
      //printf("end getRandomDirectionSimple\n");
    }
    //printf("start getTaupath\n");
    getTaupath(&d_photons[tid], &d_photons[tid].state);
    //printf("start walkNextEvent, onGrid=%d\n",d_photons[tid].onGrid);
    walkNextEvent(d_photons, d_stars, d_dustDensity, d_dustOpacity, d_grid, d_dustTemperature);//,cellWallsX, cellWallsY, cellWallsZ
    //walkNextEvent(d_photons, d_stars, d_dustDensity, d_dustOpacity, d_grid, d_dustTemperature);
    //printf("onGrid? %d\n",d_photons[tid].onGrid);
  }
}
