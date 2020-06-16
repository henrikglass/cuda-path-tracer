# CUDA directory:
CUDA_ROOT_DIR = /usr/local/cuda

# CC compiler options:
CC			= g++ #clang
CC_FLAGS	= -std=c++11 -O2 -Wall -pedantic  -p -pg
CC_LIBS		=

# NVCC compiler options:
NVCC 		= nvcc
NVCC_FLAGS 	= -O2 -dc -arch compute_75 -Xcompiler -Wall -Xcompiler -Wextra -Xcompiler -p -Xcompiler -pg  -G -lineinfo
NVCC_LIBS 	= -Iexternal

# Linker options
LINKER 		 = nvcc #g++ #clang
LINKER_FLAGS = -G -lineinfo #-std=c++11 -O2 #-Wall -pedantic
LINKER_LIBS	 =

# cuda stuff
CUDA_LIB_DIR 	= -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR 	= -I$(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS	= #-lcudart

TARGET    = cudaPathTracer
SRC_DIR   = src
OBJ_DIR   = obj

CPP_FILES = $(wildcard $(SRC_DIR)/*.cpp)
CU_FILES  = $(wildcard $(SRC_DIR)/*.cu)

H_FILES   = $(wildcard $(SRC_DIR)/*.h)
CUH_FILES = $(wildcard $(SRC_DIR)/*.cuh)

OBJ_FILES = $(addprefix $(OBJ_DIR)/,$(notdir $(CPP_FILES:.cpp=.o)))
CUO_FILES = $(addprefix $(OBJ_DIR)/,$(notdir $(CU_FILES:.cu=.cu.o)))

OBJS =  $(patsubst %.cpp,$(OBJ_DIR)/%.o,$(notdir $(CPP_FILES)))
OBJS += $(patsubst %.cu,$(OBJ_DIR)/%.cu.o,$(notdir $(CU_FILES)))

build: $(OBJ_DIR) $(TARGET)

$(TARGET): $(OBJS)
	$(LINKER) $(LINKER_FLAGS) $? -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(LINKER_LIBS)

$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu $(CUH_FILES)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(H_FILES)
	$(CC) $(CC_FLAGS) -c $< -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS) $(CC_LIBS)

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

clean:
	$(RM) output.ppm obj/* *.o $(TARGET)
