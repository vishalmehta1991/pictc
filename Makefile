INCLUDES=${CUDA_HOME}/include
HOST_COMPILER=g++
LINKER=-L${CUDA_HOME}/lib64 
COMPILER=${CUDA_HOME}/bin/nvcc
FLAGS=-arch=sm_70 -Xcompiler=-fPIC -O3 -DTENSORCORE 
NOTFLAGS=-arch=sm_70 -Xcompiler=-fPIC -O3  
all: tensor notensor

tensor: half.cpp kernelcell.cu 
	${COMPILER} -ccbin=${HOST_COMPILER} -c -o half.o half.cpp ${FLAGS}
	${COMPILER} -ccbin=${HOST_COMPILER} -c -o kernel.o kernelcell.cu  ${FLAGS}
	${COMPILER} -ccbin=${HOST_COMPILER} -o $@ *.o ${FLAGS} ${LINKER}

notensor: half.cpp kernelcell.cu 
	${COMPILER} -ccbin=${HOST_COMPILER} -c -o half.o half.cpp ${NOTFLAGS}
	${COMPILER} -ccbin=${HOST_COMPILER} -c -o kernel.o kernelcell.cu  ${NOTFLAGS}
	${COMPILER} -ccbin=${HOST_COMPILER} -o $@ *.o ${NOTFLAGS} ${LINKER}

clean:
	rm -f tensor *.o notensor
