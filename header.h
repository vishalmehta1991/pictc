
#ifndef __HALF_HEADER__
#define __HALF_HEADER__

struct particle
{
  float x,y,z;
  float vx,vy,vz;
  float vmag;
  float c,m;
};
struct particlearray
{
  float *x, *y, *z;
  float *vx, *vy, *vz;
  float *vmag;
  float *c, *m;
};
struct magneticF
{
  float mag;
  float x,y,z;
};
struct Cell
{
  float minx,miny,minz;
  float maxx,maxy,maxz;
  float Bx[8],By[8],Bz[8];
  float Bmag[8];
  
};

struct Shape
{
  float comp[15];
};

struct Deposition
{
  float w[125*3];
};
void update_position(particle& p,magneticF& b);
void update_position_borris(particle& p,magneticF& b);
void init_system(particle& P1,magneticF& B);
void update_position_gpu(Cell *cell,particlearray& p,unsigned int Nparticles,float delt,int Nts,float *outx,float *outy,float *outz);
void init_cell(Cell &cell,magneticF &field);
void deposit_charge_gpu(Deposition *depo,Shape *s0,Shape *s1,unsigned int Npar,int Nts);
// Define some error checking macros.
#define cudaErrCheck(stat) { cudaErrCheck_((stat), __FILE__, __LINE__); }
extern "C" inline void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
   if (stat != cudaSuccess) {
      fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
   }
}


#endif
