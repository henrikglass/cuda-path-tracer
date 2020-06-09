# CUDA directory:
CUDA_ROOT_DIR = /usr/local/cuda

# CC compiler options:
CC			= g++ #clang
CC_FLAGS	= -std=c++11 -O2 #-Wall -pedantic
CC_LIBS		=

# NVCC compiler options:
NVCC 		= nvcc
NVCC_FLAGS 	=
NVCC_LIBS 	=

# Linker options
LINKER 		 = g++ #clang
LINKER_FLAGS = -std=c++11 -O2 #-Wall -pedantic
LINKER_LIBS	 =

# cuda stuff
CUDA_LIB_DIR 	= -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR 	= -I$(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS	= #-lcudart

## Project file structure ##
SRC_DIR = src
OBJ_DIR = obj

# Target executable name:
TARGET = cudaPathTracer

# Object files:
SOURCES  = $(wildcard $(SRC_DIR)/*.c*)
INDLUDES = $(wildcard $(SRC_DIR)/*.*h)
OBJS 	 = $(SOURCES:wildcard $(SRC_DIR)/%.c*=$(OBJ_DIR)/%.o)

##########################################################

## Compile ##

build: $(OBJ_DIR) $(TARGET)

# Link c++ and CUDA compiled object files to target executable:
$(TARGET) : $(OBJS)
	$(LINKER) $(LINKER_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# complile main
$(OBJ_DIR)/%.o : main.cpp
	echo "compile main"
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(SRC_DIR)/%.h
	echo "compile cpp"
	$(CC) $(CC_FLAGS) -c $< -o $@

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(SRC_DIR)/%.cuh
	echo "compile cu"
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) bin/* *.o $(TARGET)