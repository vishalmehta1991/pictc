Particle Pusher In Magnetic filed using Tensor Core

This example implements a particle pusher (Borris push method) on GPUs.
The method is described here: https://www.particleincell.com/2011/vxb-rotation/

The example groups a bunch of particles that lie in a single cell and applies Magnetic field.
The electric field contribution is turned off.

The main trick here is to separate velocities and momentum into magnitude and direction. 
Magnitudes are maintained in FP32 while directions are cast to FP16.
Tensor core operates on FP16 inputs and accumulates in FP32.

Compilation:

Set CUDA_HOME

Tensor Core version: make tensor (sm_70+)

Normal Half version: make notensor


