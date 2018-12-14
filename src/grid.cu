#include <cuda_fp16.h>
#include "grid.cuh"
/*
__device__ int* getIxyzFromIndex(Grid* grid, int index){
  int* gridPosition = (int*)malloc(sizeof(int)*3);
  int nxy = grid->nCoord[0]*grid->nCoord[1];
  int iz = index/nxy;
  int idx1 = index - iz*nxy;
  int iy = idx1/grid->nCoord[0];
  int ix = idx1-iy*grid->nCoord[0];
  int* direction = (int*)malloc(sizeof(int)*3);
  gridPosition[0] = ix;
  gridPosition[1] = iy;
  gridPosition[2] = iz;
  return gridPosition;
}*/

/*
__device__ int getIndexFromIxyz(Grid* grid, int* gridPosition){
  int ix = gridPosition[0];
  int iy = gridPosition[1];
  int iz = gridPosition[2];
  return ix + iy*grid->nCoord[0] + iz*grid->nCoord[0]*grid->nCoord[1];
}*/


__device__ void getCellWalls(Photon* photon, Grid* grid, short gridPosition[], bool orientations[]){
  //axis x
  photon->cellWalls[0] = grid->cellWallsX[gridPosition[0]+orientations[0]];
  //axis y
  photon->cellWalls[1] = grid->cellWallsY[gridPosition[1]+orientations[1]];
  //axis z
  photon->cellWalls[2] = grid->cellWallsY[gridPosition[2]+orientations[2]];

}

__device__ void convertRayToGrid(Photon* photon, Grid* grid){
  //printf("IN RAY TO GRID\n");
  //ASUMIENDO NX, NY, NZ SON PARES
  int nx = grid->nCoord[0];
  int ny = grid->nCoord[1];
  int nz = grid->nCoord[2];
  double deltaX = grid->cellWallsX[1]-grid->cellWallsX[0];
  double deltaY = grid->cellWallsY[1]-grid->cellWallsY[0];
  double deltaZ = grid->cellWallsZ[1]-grid->cellWallsZ[0];

  photon->gridPosition[0] = (int) floorf(photon->rayPosition[0] / deltaX) + nx/2;
  photon->gridPosition[1] = (int) floorf(photon->rayPosition[1] / deltaY) + ny/2;
  photon->gridPosition[2] = (int) floorf(photon->rayPosition[2] / deltaZ) + nz/2;
  //printf("%d,%d,%d\n",gridPosition[0],gridPosition[1],gridPosition[2]);
}

__host__ Grid* allocateMemoryToGrid(int coordSystem, int nCoord[3]){
  Grid* grid = (Grid*) malloc(sizeof(Grid));
  grid->coordSystem = coordSystem;

  int i;
  for (i=0 ; i<3 ; i++){
    grid->nCoord[i] = nCoord[i];
  }

  grid->cellWallsX = (double*)malloc((nCoord[0]+1) * sizeof(double));
  grid->cellWallsY = (double*)malloc((nCoord[1]+1) * sizeof(double));
  grid->cellWallsZ = (double*)malloc((nCoord[2]+1) * sizeof(double));
  return grid;
}

void deallocateGrid(Grid* grid){
  free(grid->cellWallsX);
  free(grid->cellWallsY);
  free(grid->cellWallsZ);
  free(grid);
}

Grid* gridTransferToDevice(Grid* h_grid){
  printf("Transfer grid to device...\n");
  Grid* d_grid;
  double* cellWallsX;
  double* cellWallsY;
  double* cellWallsZ;
  size_t sizeWallsX = sizeof(double) * (h_grid->nCoord[0]+1);
  size_t sizeWallsY = sizeof(double) * (h_grid->nCoord[1]+1);
  size_t sizeWallsZ = sizeof(double) * (h_grid->nCoord[2]+1);


  cudaMalloc( (void**)&(d_grid), sizeof(Grid) );
  cudaMalloc((void**)&cellWallsX, sizeWallsX);
  cudaMalloc((void**)&cellWallsY, sizeWallsY);
  cudaMalloc((void**)&cellWallsZ, sizeWallsZ);

  cudaMemcpy(d_grid, h_grid, sizeof(Grid), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_grid->cellWallsX), &cellWallsX, sizeof(double *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_grid->cellWallsY), &cellWallsY, sizeof(double *), cudaMemcpyHostToDevice);
  cudaMemcpy(&(d_grid->cellWallsZ), &cellWallsZ, sizeof(double *), cudaMemcpyHostToDevice);

  cudaMemcpy(cellWallsX, h_grid->cellWallsX, sizeWallsX, cudaMemcpyHostToDevice);
  cudaMemcpy(cellWallsY, h_grid->cellWallsY, sizeWallsY, cudaMemcpyHostToDevice);
  cudaMemcpy(cellWallsZ, h_grid->cellWallsZ, sizeWallsZ, cudaMemcpyHostToDevice);

  return d_grid;
}
/*
__global__ void firstKernel(Grid* grid){
  printf("IN MYKERNEL\n");
  int i = threadIdx.x;
  printf("threadIdx%d\n",i);
  printf("coordX= %d, coordY= %d, coordZ= %d\n",grid->nCoord[0],grid->nCoord[1],grid->nCoord[2]);
  printf("cellWallsX[0]=%lf\n",grid->cellWallsX[0]);
  printf("cellWallsX[1]=%lf\n",grid->cellWallsX[1]);
}*/

Grid* readAmrGrid(){
  Grid* grid;
  char line[30];
  int coordSystem;
  int nCoord[3];
  const char* nameFile = "inputs/amr_grid.inp";
  FILE* gridFile = fopen(nameFile, "r");
  if (gridFile == NULL){
    printf("Failed to open amr_grid.inp. Maybe don't exist\n");
    exit(1);
  }else{
    fscanf(gridFile, "%d\n", &coordSystem);
    //printf("CoordSystem= %d\n",coordSystem);
    fscanf(gridFile, "%d %d %d\n", &nCoord[0],&nCoord[1],&nCoord[2]);
    //printf("nCoord= %d,%d,%d\n",nCoord[0],nCoord[1],nCoord[2]);
    if (nCoord[0] < 1 || nCoord[1] < 1 || nCoord[2] < 1){
      printf("ERROR: nx, ny, nz must be > 0");
      exit(1);
    }
    grid = allocateMemoryToGrid(coordSystem, nCoord);
    int i;
    for (i=0 ; i<nCoord[0]+1;i++){
      fgets(line, 30, gridFile);
      grid->cellWallsX[i] = atof(line);
      //printf("cellWallsX[%d] = %lf\n",i,grid->cellWallsX[i]);
      //fscanf(gridFile, "%f", &grid->cellWallsX[i]);
    }
    for (i=0 ; i<nCoord[1]+1;i++){
      fgets(line, 30, gridFile);
      grid->cellWallsY[i] = atof(line);
      //printf("cellWallsY[%d] = %lf\n",i,grid->cellWallsY[i]);
      //fscanf(gridFile, "%f", &grid->cellWallsY[i]);
    }
    for (i=0 ; i<nCoord[2]+1;i++){
      fgets(line, 30, gridFile);
      grid->cellWallsZ[i] = atof(line);
      //printf("cellWallsZ[%d] = %lf\n",i,grid->cellWallsZ[i]);
      //fscanf(gridFile, "%f", &grid->cellWallsZ[i]);
    }
    fclose(gridFile);
    return grid;
  }
}

void calculateCellVolumes(Grid* grid){
  double deltaX = grid->cellWallsX[1] - grid->cellWallsX[0];
  double deltaY = grid->cellWallsY[1] - grid->cellWallsY[0];
  double deltaZ = grid->cellWallsZ[1] - grid->cellWallsZ[0];
  grid->cellVolumes = deltaX * deltaY * deltaZ;
}

Grid* setUpGrid(){
  printf("Set up grid...\n");
  Grid* grid = readAmrGrid();
  calculateCellVolumes(grid);
  grid->totalCells = grid->nCoord[0]*grid->nCoord[1]*grid->nCoord[2];
  return grid;
}
