.SUFFIXES : .cu 

SHELL := /bin/bash

NVCC_PATH := /usr/local/cuda-5.0/

CC	:= gcc
CXX 	:= g++
NVCC	:= $(NVCC_PATH)/bin/nvcc

CXX_DIR := /home/cs-user/blast/src/ncbi-blast-2.2.26+-src/c++
CUDA_LIB := $(NVCC_PATH)/lib64

PROJ_DIR := /home/cs-user/blast/src/ncbi-blast-2.2.26+-src/c++/GCC447-ReleaseMT64

CPP_SRC = gpu_logfile.cpp gpu_blastn_na_ungapped_v3.cpp gpu_blastn_pre_search_engine.cpp gpu_blast_multi_gpu_utils.cpp work_thread.cpp work_thread_base.cpp thread_work_queue.cpp
OBJ = $(CPP_SRC:.cpp=.o)

#@echo $(OBJ)

INCLUDE = -I$(CXX_DIR)/include/ -I$(PROJ_DIR)/inc/ -I$(CXX_DIR)/src -I./ -I$(NVCC_PATH)/include -I$(NVCC_PATH)/samples/common/inc 

COMMONFLAGS := -c -D_LINUX
OPT_COMMAND := -mssse3
# the "--no-align-double" is necessary. Without it the nvcc compiler aligns structure sizes every 8 bytes which is 
#incompatible with some NCBI_BLAST data structures, i.e. the "BlastContextInfo"
NVCCFLAGS :=  -O3 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35 -Xptxas -v --no-align-double

all: libgpublastn.a

libgpublastn.a : $(OBJ) gpu_blastn_MB_and_smallNa.o 
	ar cru libgpublastn.a $(OBJ) gpu_blastn_MB_and_smallNa.o
	ranlib libgpublastn.a	
	cp libgpublastn.a                          $(PROJ_DIR)/lib/
	cp $(CUDA_LIB)/libcudart.so               $(PROJ_DIR)/lib/

%.o: %.cpp
	$(CXX) $(COMMONFLAGS) $(OPT_COMMAND) $(INCLUDE) -c -o $@ $<

gpu_blastn_MB_and_smallNa.o : 
	$(NVCC) $(COMMONFLAGS) $(NVCCFLAGS) $(INCLUDE) gpu_blastn_MB_and_smallNa.cu  

clean:
	rm *.o *.a
