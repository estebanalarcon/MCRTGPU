#ifndef GRID_CUH
#define GRID_CUH

#include "structs.cuh"

__device__ void getCellWalls(Photon* photon, Grid* grid, short gridPosition[], bool orientations[]);
__device__ void convertRayToGrid(Photon* photon, Grid* grid);
__host__ Grid* allocateMemoryToGrid(int coordSystem, int nCoord[3]);
__host__ void deallocateGrid(Grid* grid);
__host__ Grid* gridTransferToDevice(Grid* h_grid);
Grid* readAmrGrid();
void calculateCellVolumes(Grid* grid);
Grid* setUpGrid();

#endif
