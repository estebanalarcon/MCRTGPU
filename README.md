# MCRTGPU: MONTE CARLO RADIATIVE TRANSFER FOR GPU
## Authors: 
E. Alarcón
F.R. Rannou
P.E. Román

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

