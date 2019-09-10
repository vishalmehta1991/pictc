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
#ifndef __KERNELCELL__
#define __KERNELCELL__

#include "header.h"
#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <stdio.h>
#include <stdlib.h>

#define PARPERWARP 32
namespace cg = cooperative_groups;

__global__ void update_position_gpu_kernel_cell(
    const Cell *cell, particlearray P, const __restrict__ half *Tmat,
    const __restrict__ half *Smat, const unsigned int Nparticles, const int Nts,
    const float delt, float *outx, float *outy, float *outz);
__global__ void deposit_charge_gpu_kernel(Deposition *depo, Shape *s0,
                                          Shape *s1, unsigned int Npar,
                                          int Nts);
void calculate_TmatSmat(half *Tmat, half *Smat, Cell &cell, float delt);
__device__ inline void dumpaccmat(__half *accmat, float vx, float vy, float vz,
                                  int tid);
__device__ inline void fillpmat(__half *Pmat, float vx, float vy, float vz,
                                int tid);
__device__ void matmultid(__half *Pmat, __half *Bmat, float *accmat);
__device__ inline void matrixvec_tensor(half *a, half *b, float *c);

#endif
