#ifndef __KERNELCELL__
#define __KERNELCELL__

#include <stdio.h>
#include <stdlib.h>
#include "header.h"
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>


#define PARPERWARP 32
namespace cg = cooperative_groups;

__global__ void update_position_gpu_kernel_cell(const Cell *cell,particlearray P,const __restrict__ half *Tmat,const __restrict__ half *Smat,const unsigned int Nparticles,const int Nts,const float delt,float *outx,float *outy,float *outz);
__global__ void deposit_charge_gpu_kernel(Deposition *depo,Shape *s0,Shape *s1,unsigned int Npar,int Nts);
void calculate_TmatSmat(half *Tmat,half *Smat,Cell &cell,float delt);
__device__ inline void dumpaccmat(__half *accmat,float vx,float vy,float vz,int tid);
__device__ inline void fillpmat(__half *Pmat,float vx,float vy,float vz,int tid);
__device__ void matmultid(__half *Pmat,__half *Bmat,float *accmat);
__device__ inline void matrixvec_tensor(half *a, half *b, float *c);

#endif
