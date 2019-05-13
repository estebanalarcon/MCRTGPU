# MCRTGPU: MONTE CARLO RADIATIVE TRANSFER FOR GPU
## Contributors: 
E. Alarcón - Universidad de Santiago de Chile esteban.alarcon.v@usach.cl

F.R. Rannou - Universidad de Santiago de Chile

P.E. Román - Universidad de Santiago de Chile

## Installation
1.- Install CUDA >= 9.

2.- Download or clone MCRTGPU.

## Compiling
```bash
cd MCRTGPU
make
```

## Execution parameters
-N: Number of total photons. It must be multiple of 1024.

-n: Number of parallel photons to exec. it must be less or equal to N, and divisor to N. Default = 51.200 (1024 x 50)

-t: (opcional) Number of temperatures in range. Default = 1000

-i: (opcional) Initial temperature. Default = 0.01

-f: (opcional) Final temperature. Default = 100000

-s: Scattering mode.
* 0: No Scattering
* 1: Isotropic Scattering
* 2: Anisotorpic Scattering

-b: CUDA block size. It must be 2,4,8,16,32,64,128,256 or 512. 32 is recommended.

## Usage

1. Create inputs/ folder in MCRTGPU directory.
2. Copy input files in new folder.
3. In MCRTGPU directory, run command (example):
```bash
./bin/mcrtgpu -N 102400 -b 32 -s 0
```
4. The out file name is dust_temperature_gpu.dat.

Important: The input files are the same format that RADMC-3D. Warning with spaces characters between values in the same line: It must be one space character or one tab character. Other way would cause incorrect results. 
