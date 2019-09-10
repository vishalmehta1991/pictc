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
#ifndef __HALF_HEADER__
#define __HALF_HEADER__

struct particle {
  float x, y, z;
  float vx, vy, vz;
  float vmag;
  float c, m;
};
struct particlearray {
  float *x, *y, *z;
  float *vx, *vy, *vz;
  float *vmag;
  float *c, *m;
};
struct magneticF {
  float mag;
  float x, y, z;
};
struct Cell {
  float minx, miny, minz;
  float maxx, maxy, maxz;
  float Bx[8], By[8], Bz[8];
  float Bmag[8];
};

struct Shape {
  float comp[15];
};

struct Deposition {
  float w[125 * 3];
};
void update_position(particle &p, magneticF &b);
void update_position_borris(particle &p, magneticF &b);
void init_system(particle &P1, magneticF &B);
void update_position_gpu(Cell *cell, particlearray &p, unsigned int Nparticles,
                         float delt, int Nts, float *outx, float *outy,
                         float *outz);
void init_cell(Cell &cell, magneticF &field);
void deposit_charge_gpu(Deposition *depo, Shape *s0, Shape *s1,
                        unsigned int Npar, int Nts);
// Define some error checking macros.
#define cudaErrCheck(stat)                                                     \
  { cudaErrCheck_((stat), __FILE__, __LINE__); }
extern "C" inline void cudaErrCheck_(cudaError_t stat, const char *file,
                                     int line) {
  if (stat != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file,
            line);
  }
}

#endif
