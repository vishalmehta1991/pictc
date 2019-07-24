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
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include "libhalf/include/half.hpp"
#include <iostream>
#include <cuda_runtime_api.h>
#include "header.h"
      

//#define Npar 32
#define Npar 51200
#define datatype float
using half_float::half;
float delt = 0.001;

int main( int argc, char* argv[] )
{
  particle *pp = (particle*)malloc(sizeof(particle)*Npar);
  particlearray h_p,d_p;
  int nsteps = 2000;
  particle &P1 = pp[0];
  magneticF B;
  particle *d_P;
  magneticF *d_b;
  float *outx,*outy,*outz;
  float *d_outx,*d_outy,*d_outz;
  Cell cell,*d_cell;
  Shape *d_s0, *d_s1;
  Deposition *d_depo;
  
  outx = (float*)malloc(nsteps*sizeof(float));
  outy = (float*)malloc(nsteps*sizeof(float));
  outz = (float*)malloc(nsteps*sizeof(float));
  h_p.x = (float*)malloc(Npar*sizeof(float));
  h_p.y = (float*)malloc(Npar*sizeof(float));
  h_p.z = (float*)malloc(Npar*sizeof(float));
  h_p.vx= (float*)malloc(Npar*sizeof(float));
  h_p.vy= (float*)malloc(Npar*sizeof(float));
  h_p.vz= (float*)malloc(Npar*sizeof(float));
  h_p.vmag = (float*)malloc(Npar*sizeof(float));
  
  
  for(int i=0;i<Npar;i++)
    {
      particle &P = pp[i];
      init_system(P,B);
      h_p.x[i] = P.x;      h_p.y[i] = P.y;      h_p.z[i] = P.z;
      h_p.vx[i] = P.vx;      h_p.vy[i] = P.vy;      h_p.vz[i] = P.vz;
      h_p.vmag[i] = P.vmag;
    }
  init_cell(cell,B);
  
  cudaErrCheck(cudaDeviceReset());
  cudaErrCheck(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte ));
  cudaErrCheck(cudaMalloc((void**)(&d_cell),sizeof(Cell)));
  cudaErrCheck(cudaMalloc((void**)(&(d_p.x)),Npar*sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)(&(d_p.y)),Npar*sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)(&(d_p.z)),Npar*sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)(&(d_p.vx)),Npar*sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)(&(d_p.vy)),Npar*sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)(&(d_p.vz)),Npar*sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)(&(d_p.vmag)),Npar*sizeof(float)));

  cudaErrCheck(cudaMalloc((void**)(&(d_s0)),Npar*sizeof(Shape)));
  cudaErrCheck(cudaMalloc((void**)(&(d_s1)),Npar*sizeof(Shape)));
  cudaErrCheck(cudaMalloc((void**)(&(d_depo)),Npar*sizeof(Deposition)));
  
  cudaErrCheck(cudaMalloc((void**)(&d_P),Npar*sizeof(particle)));
  cudaErrCheck(cudaMalloc((void**)(&d_b),sizeof(magneticF)));
  cudaErrCheck(cudaMalloc((void**)(&d_outx),nsteps*sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)(&d_outy),nsteps*sizeof(float)));
  cudaErrCheck(cudaMalloc((void**)(&d_outz),nsteps*sizeof(float)));
  cudaErrCheck(cudaMemcpy(d_P,pp,Npar*sizeof(particle),cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_b,&B,sizeof(magneticF),cudaMemcpyHostToDevice));
  
  cudaErrCheck(cudaMemcpy(d_cell,&cell,sizeof(Cell),cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_p.x,h_p.x,Npar*sizeof(float),cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_p.y,h_p.y,Npar*sizeof(float),cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_p.z,h_p.z,Npar*sizeof(float),cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_p.vx,h_p.vx,Npar*sizeof(float),cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_p.vy,h_p.vy,Npar*sizeof(float),cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_p.vz,h_p.vz,Npar*sizeof(float),cudaMemcpyHostToDevice));
  cudaErrCheck(cudaMemcpy(d_p.vmag,h_p.vmag,Npar*sizeof(float),cudaMemcpyHostToDevice));

  update_position_gpu(d_cell,d_p,Npar,delt,nsteps,d_outx,d_outy,d_outz);

  cudaErrCheck(cudaMemcpy(outx,d_outx,nsteps*sizeof(float),cudaMemcpyDeviceToHost ));
  cudaErrCheck(cudaMemcpy(outy,d_outy,nsteps*sizeof(float),cudaMemcpyDeviceToHost ));
  cudaErrCheck(cudaMemcpy(outz,d_outz,nsteps*sizeof(float),cudaMemcpyDeviceToHost ));

  cudaErrCheck(cudaDeviceSynchronize());
  
  cudaErrCheck(cudaDeviceReset());
  /*  FILE *g = fopen("outputgpu.bin","wb");

  for(int i=0;i<nsteps;i++)
    {
      float bb[3];  
      bb[0] = outx[i]; bb[1] = outy[i]; bb[2] = outz[i]; 
      fwrite(bb,sizeof(bb),1,g);
    }
  fclose(g);
  */
  return 0;
}

void update_position_borris(particle& p,magneticF& b)
{
  //The electric filed is put to zero. Edit the code to add it.
  datatype vminus[3],vprime[3],tvec[3],svec[3];
  float tscal[3];
  vminus[0] = (datatype)p.vx; vminus[1] = (datatype)p.vy; vminus[2] = (datatype)p.vz;

  float fac = (p.c*delt*b.mag)/(2*p.m);
  tscal[0] = fac*b.x; tscal[1] = fac*b.y; tscal[2] = fac*b.z;
  float tmag = sqrt(tscal[0]*tscal[0] + tscal[1]*tscal[1] + tscal[2]*tscal[2]);
  tmag = tmag+1;
  tmag = 2/tmag;
  
  tvec[0] = (datatype)tscal[0];
  tvec[1] = (datatype)tscal[1];
  tvec[2] = (datatype)tscal[2];

  tscal[0] *=tmag;  tscal[1] *=tmag;  tscal[2] *=tmag;
  svec[0] = (datatype)tscal[0];
  svec[1] = (datatype)tscal[1];
  svec[2] = (datatype)tscal[2];

  
  datatype mat[9];
  mat[0]=(datatype)0;mat[4]=(datatype)0;mat[8]=(datatype)0;
  mat[1]=(datatype)(-1)*vminus[2];
  mat[2]=vminus[1];
  mat[3]=vminus[2];
  mat[5]=(datatype)(-1)*vminus[0];
  mat[6]=(datatype)(-1)*vminus[1];
  mat[7]=vminus[0];

  float acc[3];
  acc[0] = mat[0] * tvec[0] + mat[1] * tvec[1] + mat[2] * tvec[2];
  acc[1] = mat[3] * tvec[0] + mat[4] * tvec[1] + mat[5] * tvec[2];
  acc[2] = mat[6] * tvec[0] + mat[7] * tvec[1] + mat[8] * tvec[2];

  acc[0] = vminus[0] + acc[0]*p.vmag;
  acc[1] = vminus[1] + acc[1]*p.vmag;
  acc[2] = vminus[2] + acc[2]*p.vmag;
  float vprimemag = sqrt(acc[0]*acc[0] + acc[1]*acc[1] + acc[2]*acc[2]);
  acc[0] /= vprimemag; acc[1] /= vprimemag; acc[2] /= vprimemag;
  
  vprime[0] = (datatype)acc[0];
  vprime[1] = (datatype)acc[1];
  vprime[2] = (datatype)acc[2];
  mat[1]=(datatype)(-1)*vprime[2];
  mat[2]=vprime[1];
  mat[3]=vprime[2];
  mat[5]=(datatype)(-1)*vprime[0];
  mat[6]=(datatype)(-1)*vprime[1];
  mat[7]=vprime[0];

  acc[0] = mat[0] * svec[0] + mat[1] * svec[1] + mat[2] * svec[2];
  acc[1] = mat[3] * svec[0] + mat[4] * svec[1] + mat[5] * svec[2];
  acc[2] = mat[6] * svec[0] + mat[7] * svec[1] + mat[8] * svec[2];
  acc[0] = vminus[0] + acc[0]*vprimemag;
  acc[1] = vminus[1] + acc[1]*vprimemag;
  acc[2] = vminus[2] + acc[2]*vprimemag;

  p.x += acc[0]*delt;
  p.y += acc[1]*delt;
  p.z += acc[2]*delt;

  p.vmag = sqrt(acc[0]*acc[0] + acc[1]*acc[1] + acc[2]*acc[2]);
  p.vx = (acc[0]/p.vmag);
  p.vy = (acc[1]/p.vmag);
  p.vz = (acc[2]/p.vmag);
  
  return;
}
void update_position_implicit(particle& p,magneticF& b)
{
  //There is no electric field in this solution
  float fac = (p.c * b.mag * delt)/(2*p.m); //this is eps constant
  float bx = (fac*b.x)/b.mag;
  float by = (fac*b.y)/b.mag;
  float bz = (fac*b.z)/b.mag;

  //determinant & inverse of [I - R * eps] 
  float detb = (1+ bx*bx) + bz * (bz - bx*by) + by * (bz*bx + by);
  datatype invmat[9];
  invmat[0] = (datatype)((1 + fac*fac*b.x*b.x)/detb);
  invmat[1] = (datatype)((fac*b.z + fac*fac*b.x*b.y)/detb);
  invmat[2] = (datatype)((-1*fac*b.y + fac*fac*b.x*b.z)/detb);
  invmat[3] = (datatype)((-1*fac*b.z + fac*fac*b.x*b.y)/detb);
  invmat[4] = (datatype)((1 + fac*fac*b.y*b.y)/detb);
  invmat[5] = (datatype)((fac*b.x + fac*fac*b.y*b.z)/detb);
  invmat[6] = (datatype)((fac*b.y + fac*fac*b.x*b.z)/detb);
  invmat[7] = (datatype)((-1*fac*b.x + fac*fac*b.y*b.z)/detb);
  invmat[0] = (datatype)((1 + fac*fac*b.z*b.z)/detb);

  //matrix [I + R*eps]
  datatype mat[9];
  mat[0] = (datatype)(1.0);
  mat[1] = (datatype)(bz);
  mat[2] = (datatype)(-1*by);
  mat[3] = (datatype)(-1*bz);
  mat[4] = (datatype)(1.0);
  mat[5] = (datatype)(bx);
  mat[6] = (datatype)(by);
  mat[7] = (datatype)(-1*bx);
  mat[8] = (datatype)(1);

  //Here we update the postion as per v(1) = inv([I - R*eps])*([I + R*eps])*v[0]
  datatype vx,vy,vz;
  vx = mat[0] * p.vx + mat[1] * p.vy + mat[2] * p.vz;
  vy = mat[3] * p.vx + mat[4] * p.vy + mat[5] * p.vz;
  vz = mat[6] * p.vx + mat[7] * p.vy + mat[8] * p.vz;

  p.vx = invmat[0] * vx + invmat[1] * vy + invmat[2] * vz;
  p.vy = invmat[3] * vx + invmat[4] * vy + invmat[5] * vz;
  p.vz = invmat[6] * vx + invmat[7] * vy + invmat[8] * vz;

  float vxx,vyy,vzz;
  vxx = p.vmag*((float)p.vx); vyy = p.vmag*((float)p.vy); vzz = p.vmag*((float)p.vz);
  p.x += vxx*delt;
  p.y += vyy*delt;
  p.z += vzz*delt;

  p.vmag = sqrt(vxx*vxx + vyy*vyy + vzz*vzz);
  vxx /= p.vmag; vyy /= p.vmag; vzz /= p.vmag;
  p.vx = (datatype)vxx;  p.vy = (datatype)vyy;  p.vz = (datatype)vzz;
  
  return;
}

void init_system(particle& P1,magneticF& B)
{
  P1.x = 0.0;
  P1.y = 0.0;
  P1.z = 0.0;

  P1.c = 1.0;
  P1.m = 1.0;
  
  P1.vx = 0;
  P1.vy = 1.0;
  P1.vz = 0;
  P1.vmag = 1.0; //sqrt(P1.vx*P1.vx + P1.vy*P1.vy + P1.vz*P1.vz);
  
  B.mag = 10.0;
  B.x = 0;
  B.y = 0;
  B.z = 1;
  return;
}

void init_cell(Cell &cell,magneticF &field)
{
  cell.minx = -2;
  cell.miny = -2;
  cell.minz = -2;
  cell.maxx = 2;
  cell.maxy = 2;
  cell.maxz = 2;
  
  for(int i=0;i<8;i++)
    {
      cell.Bx[i] = field.x;cell.By[i] = field.y;cell.Bz[i] = field.z;
      cell.Bmag[i] = field.mag;
    }
  return;
}
