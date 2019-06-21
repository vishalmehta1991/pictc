INCLUDES=${CUDA_HOME}/include
HOST_COMPILER=g++
LINKER=-L${CUDA_HOME}/lib64 
COMPILER=${CUDA_HOME}/bin/nvcc
FLAGS=-arch=sm_70 -Xcompiler=-fPIC -O3 -DTENSORCORE 
NOTFLAGS=-arch=sm_70 -Xcompiler=-fPIC -O3  
all: half notensor

half: half.cpp kernelcell.cu 
	${COMPILER} -ccbin=${HOST_COMPILER} -c -o half.o half.cpp ${FLAGS}
	${COMPILER} -ccbin=${HOST_COMPILER} -c -o kernel.o kernelcell.cu  ${FLAGS}
	${COMPILER} -ccbin=${HOST_COMPILER} -o $@ *.o ${FLAGS} ${LINKER}

notensor: half.cpp kernelcell.cu 
	${COMPILER} -ccbin=${HOST_COMPILER} -c -o half.o half.cpp ${NOTFLAGS}
	${COMPILER} -ccbin=${HOST_COMPILER} -c -o kernel.o kernelcell.cu  ${NOTFLAGS}
	${COMPILER} -ccbin=${HOST_COMPILER} -o $@ *.o ${NOTFLAGS} ${LINKER}

clean:
	rm -f main *~ half *.o *.bin skew *.nvprof notensor

profile: half
	nvprof -m shared_efficiency -e shared_load,shared_store,shared_ld_bank_conflict,shared_st_bank_conflict,shared_ld_transactions,shared_st_transactions ./half
test: some.cu
	nvcc -arch=sm_70 -o skew some.cu && nvprof -m shared_efficiency -e shared_load,shared_store,shared_ld_bank_conflict,shared_st_bank_conflict,shared_ld_transactions,shared_st_transactions ./skew

