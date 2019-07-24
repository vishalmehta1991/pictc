/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include "kernelcell.h"
#define SKEW 0
#define SKEW2 0
#define NWARPS 2

//Matrix vector product warp level using Tensor Core CUDA wmma API
__device__ __forceinline__ void matrixvec_tensor(half *a, half *b, float *c) 
{
  
  // The only dimensions currently supported by WMMA
  // Declare the fragments
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half, nvcuda::wmma::col_major> a_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half, nvcuda::wmma::row_major> b_frag;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, float> acc_frag;
  nvcuda::wmma::fill_fragment(acc_frag, 0.0f);  
  nvcuda::wmma::load_matrix_sync(a_frag, a, 8);
  nvcuda::wmma::load_matrix_sync(b_frag, b, 32);
  
  nvcuda::wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  
  nvcuda::wmma::store_matrix_sync(c , acc_frag, 32, nvcuda::wmma::mem_row_major);
  return;
}

__device__ __forceinline__ void calculate_scale(float px,float py,float pz,float minx,float miny,float minz,float maxx,float maxy,float maxz,float& alpha,float& beta,float& gamma)
{
  //alpha*higher + (1-alpha)*lower
  alpha = (maxx - px) / (maxx - minx);
  beta = (maxy - py) / (maxy - miny);
  gamma = (maxz - pz) / (maxz - minz);
  return;
}
__device__ __forceinline__ void fillpmat(half *Pmat,float vx,float vy,float vz,int tid,float alpha,float beta)
{
  int colid = (int)(tid%32);
  float scale = (1-alpha)*(1-beta);
  Pmat[colid     ] = __float2half(scale*vx);
  Pmat[colid + 32] = __float2half(scale*vy);
  Pmat[colid + 64] = __float2half(scale*vz);  

  colid += 96;
  scale = alpha*(1-beta);
  Pmat[colid ] = __float2half(scale*vx);
  Pmat[colid + 32] = __float2half(scale*vy);
  Pmat[colid + 64] = __float2half(scale*vz);  

  colid += 96;
  scale = (alpha)*beta;
  Pmat[colid ] = __float2half(scale*vx);
  Pmat[colid + 32] = __float2half(scale*vy);
  Pmat[colid + 64] = __float2half(scale*vz);  
  
  colid += 96;
  scale = (1-alpha)*beta;
  Pmat[colid ] = __float2half(scale*vx);
  Pmat[colid + 32] = __float2half(scale*vy);
  Pmat[colid + 64] = __float2half(scale*vz);  
  
  colid += 96;
  Pmat[colid ] = __float2half(0.0);
  Pmat[colid + 32] = __float2half(0.0);
  Pmat[colid + 64] = __float2half(0.0);  
  Pmat[colid + 96] = __float2half(0.0);
  return;
}
__device__ __forceinline__ void dumpaccmat(float *accmat,float &vx,float &vy,float &vz,int tid,float gamma)
{
  int colid = (int)(tid%32);
  vx = (1-gamma)*accmat[colid     ] + gamma*accmat[colid + 96];
  vy = (1-gamma)*accmat[colid + 16] + gamma*accmat[colid + 96 + 16];
  vz = (1-gamma)*accmat[colid + 32] + gamma*accmat[colid + 96 + 32];   
  return;
}

__device__ __forceinline__ void matmultid(half *X,half *Y,float *accmat,int tid)
{
  int id = (int)(threadIdx.x % 32);
  
  for(int k = 0;k<8;k++)
    {
      float t =0; 
      for(int i =0;i<16;i++)
	{
	  t += __half2float( __hmul(X[i + k*16] , Y[id + i * 32]) ); 
	}
      accmat[id + k*32] = t;
    }

  return;
}

//Main driver function to call GPU kernels
void update_position_gpu(Cell *cell,particlearray& p,unsigned int Nparticles,float delt,int Nts,float *outx,float *outy,float *outz)
{
  half *Tmat,*Smat;
  unsigned int nsz = 8*16;
  half *h_Tmat,*h_Smat;
  Cell h_cell;
  
  h_Tmat = (half*)malloc(nsz*sizeof(half));
  h_Smat = (half*)malloc(nsz*sizeof(half));
  memset((void*)h_Tmat,0,nsz*sizeof(half));
  memset((void*)h_Smat,0,nsz*sizeof(half));

  cudaErrCheck(cudaMemcpy(&h_cell,cell,sizeof(Cell),cudaMemcpyDeviceToHost));
  calculate_TmatSmat(h_Tmat,h_Smat,h_cell,delt);
  
  cudaErrCheck(cudaMalloc((void**)(&Tmat),nsz*sizeof(half)));
  cudaErrCheck(cudaMalloc((void**)(&Smat),nsz*sizeof(half)));
  cudaErrCheck(cudaMemcpy(Tmat,h_Tmat,nsz*sizeof(half),cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(Smat,h_Smat,nsz*sizeof(half),cudaMemcpyHostToDevice));
 
  unsigned int nblocks = ceil( (Nparticles/PARPERWARP)/NWARPS );
  if (nblocks == 0) nblocks =1;
  
  update_position_gpu_kernel_cell<<<nblocks,NWARPS*32>>>(cell,p,Tmat,Smat,Nparticles,Nts,delt,outx,outy,outz);
  cudaErrCheck(cudaGetLastError());
  
  cudaErrCheck(cudaFree(Tmat));
  cudaErrCheck(cudaFree(Smat));

  return;
}

//Main GPU kernel that updates particle postion using mixed precision
__global__ void update_position_gpu_kernel_cell(const Cell *cell,particlearray P,const __restrict__ half *Tmat,const __restrict__ half *Smat,const unsigned int Nparticles,const int Nts,const float delt,float *outx,float *outy,float *outz)
{
  __shared__ half shTmat[8*16],shSmat[8*16],Pmatr[16*32*NWARPS];
  __shared__ float accmatr[8*32*NWARPS];
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int warpoff = (int)(threadIdx.x / 32 );
  int warpoffp = warpoff * 16 * 32; 
  int warpoffa = warpoff * 8 * 32;

  float tx,ty,tz,vprimemag;
  float alpha,beta,gamma;
  float minx,miny,minz,maxx,maxy,maxz;

  //Load Tmat and Smat
  for(int i = threadIdx.x;i<8*16; i += blockDim.x)
    {
      shTmat[i] = Tmat[i];shSmat[i] = Smat[i];
    }

  if(tid < Nparticles)
    {  
      minx = cell->minx;miny = cell->miny;minz = cell->minz;
      maxx = cell->maxx;maxy = cell->maxy;maxz = cell->maxz;
      for(int i=0;i<Nts;i++)
	{
	  //Scale float values; in-order to work in half precision
	  calculate_scale(P.x[tid],P.y[tid],P.z[tid],minx,miny,minz,maxx,maxy,maxz,alpha,beta,gamma);
	  fillpmat(&Pmatr[warpoffp],P.vx[tid],P.vy[tid],P.vz[tid],tid,alpha,beta);

	  //Use regular matrix-vector vs Tensor Core for updating velocities.
#ifdef TENSORCORE	  
	  matrixvec_tensor(shTmat,&Pmatr[warpoffp],&accmatr[warpoffa]);
	  cg::this_thread_block().sync();
#else
	  matmultid(shTmat,&Pmatr[warpoffp],&accmatr[warpoffa],tid);
#endif
	  //Back convert and resacle
	  dumpaccmat(&accmatr[warpoffa],tx,ty,tz,tid,gamma);
	  tx *= -1; ty *= -1; tz *= -1;
	  tx = P.vx[tid] + tx * P.vmag[tid];
	  ty = P.vy[tid] + ty * P.vmag[tid];
	  tz = P.vz[tid] + tz * P.vmag[tid];
	  
	  vprimemag = sqrt(tx*tx + ty*ty + tz*tz);
	  tx /= vprimemag; ty /= vprimemag; tz /= vprimemag;
	  
	  fillpmat(&Pmatr[warpoffp],tx,ty,tz,tid,alpha,beta);

	  //Use regular matrix-vector vs Tensor Core for updating positions.
#ifdef TENSORCORE	  
	  matrixvec_tensor(shSmat,&Pmatr[warpoffp],&accmatr[warpoffa]);	 
	  cg::this_thread_block().sync();
#else
	  matmultid(shSmat,&Pmatr[warpoffp],&accmatr[warpoffa],tid);
#endif

	  dumpaccmat(&accmatr[warpoffa],tx,ty,tz,tid,gamma);
	  tx *= -1; ty *= -1; tz *= -1;
	  tx = P.vx[tid] + tx * vprimemag;
	  ty = P.vy[tid] + ty * vprimemag;
	  tz = P.vz[tid] + tz * vprimemag;	  

	  vprimemag = sqrt(tx*tx + ty*ty + tz*tz);
	  //Update particle velcities magnitude and direction
	  P.vmag[tid] = vprimemag;
	  P.vx[tid] = (tx/vprimemag);
	  P.vy[tid] = (ty/vprimemag);
	  P.vz[tid] = (tz/vprimemag);
	  
	  //Update particle positions in float
	  P.x[tid] += tx*delt;
	  P.y[tid] += ty*delt;
	  P.z[tid] += tz*delt;
	}  
    }
  return;
}

//Calculation and scaling for T matrix and S matrix; and convert to half
void calculate_TmatSmat(half *Tmat,half *Smat,Cell &cell,float delt)
{
  float tscal[3],tmag;
  float particlecharge=1.0;
  float particlemass=1.0;
  
  //ordering the matrices in column major
  for(int i=0;i<8;i++)
    {
      float fac = (particlecharge*delt*cell.Bmag[i])/(2*particlemass);
      tscal[0] = fac*cell.Bx[i]; tscal[1] = fac*cell.By[i]; tscal[2] = fac*cell.Bz[i];
      tmag = sqrt(tscal[0]*tscal[0] + tscal[1]*tscal[1] + tscal[2]*tscal[2]);
      tmag = tmag+1;
      tmag = 2/tmag;
      
      int off;
      if(i<4) 
	{
	  off = i*8*3;
	}
      else
	{
	  off = 3 + (i-4)*8*3;
	}
      
      int idx = off;
      int dsz = 8;
      Tmat[idx] = __float2half(0.0);
      Tmat[idx + 1] = __float2half(tscal[2]); 
      Tmat[idx + 2] = __float2half(-1 * tscal[1]);
      Tmat[off + dsz*1 ] = __float2half(-1 * tscal[2]);
      Tmat[off + dsz*1 + 1] = __float2half(0.0);
      Tmat[off + dsz*1 + 2] = __float2half(tscal[0]); 
      Tmat[off + dsz*2 ] = __float2half(tscal[1]); 
      Tmat[off + dsz*2 + 1] = __float2half(-1*tscal[0]);
      Tmat[off + dsz*2 + 2] = __float2half(0.0);      
      
      tscal[0] *=tmag;  tscal[1] *=tmag;  tscal[2] *=tmag;

      Smat[idx] = __float2half(0.0);
      Smat[idx + 1] = __float2half(tscal[2]); 
      Smat[idx + 2] = __float2half(-1 * tscal[1]);
      Smat[off + dsz*1 ] = __float2half(-1 * tscal[2]);
      Smat[off + dsz*1 + 1] = __float2half(0.0);
      Smat[off + dsz*1 + 2] = __float2half(tscal[0]); 
      Smat[off + dsz*2 ] = __float2half(tscal[1]); 
      Smat[off + dsz*2 + 1] = __float2half(-1*tscal[0]);
      Smat[off + dsz*2 + 2] = __float2half(0.0);      
      
    }
  return;
}

