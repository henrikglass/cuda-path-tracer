## CUDA directory:
#CUDA_ROOT_DIR = /usr/local/cuda
#
## CC compiler options:
#CC			= g++ #clang
#CC_FLAGS	= -std=c++11 -O2 #-Wall -pedantic
#CC_LIBS		=
#
## NVCC compiler options:
#NVCC 		= nvcc
#NVCC_FLAGS 	=
#NVCC_LIBS 	=
#
## Linker options
#LINKER 		 = g++ #clang
#LINKER_FLAGS = #-std=c++11 -O2 #-Wall -pedantic
#LINKER_LIBS	 =
#
## cuda stuff
#CUDA_LIB_DIR 	= -L$(CUDA_ROOT_DIR)/lib64
#CUDA_INC_DIR 	= -I$(CUDA_ROOT_DIR)/include
#CUDA_LINK_LIBS	= #-lcudart
#
### Project file structure ##
#SRC_DIR = src
#OBJ_DIR = obj
#BIN_DIR = .
#
## Target executable name:
#TARGET = cudaPathTracer
#
## Object files:
#SOURCES  = $(wildcard $(SRC_DIR)/*.c*)
#INDLUDES = $(wildcard $(SRC_DIR)/*.*h)
#OBJS 	 = $(SOURCES:wildcard $(SRC_DIR)/%.c*=$(OBJ_DIR)/%.o)
#OBJ   	 = $(wildcard $(OBJ_DIR)/*.o)
#
###########################################################
#
### Compile ##
#
#build: $(OBJ_DIR) $(TARGET)
#
#$(BIN_DIR)/$(TARGET): $(OBJS)
#	$(LINKER) $(LINKER_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)
#
#$(OBJS): $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu
#	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
#
#$(OBJS): $(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp
#	$(CC) $(CC_FLAGS) -c $< -o $@
#
#$(OBJ_DIR):
#	mkdir $(OBJ_DIR)
#
### Link c++ and CUDA compiled object files to target executable:
##$(TARGET) : $(OBJS)
##	$(LINKER) $(LINKER_FLAGS) $(OBJ) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)
##
### complile main
##$(OBJ_DIR)/%.o : main.cu
##	$(NVCC) $(NVCC_FLAGS) -c $< -o $@
##
### Compile C++ source files to object files:
##$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(SRC_DIR)/%.h
##	echo "compile cpp"
##	$(CC) $(CC_FLAGS) -c $< -o $@
##
### Compile CUDA source files to object files:
##$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(SRC_DIR)/%.cuh
##	echo "compile cu"
##	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)
#
## Clean objects in object directory.
#clean:
#	$(RM) bin/* *.o $(TARGET)

# CUDA directory:
CUDA_ROOT_DIR = /usr/local/cuda

# CC compiler options:
CC			= g++ #clang
CC_FLAGS	= -std=c++11 -O2 #-Wall -pedantic
CC_LIBS		=

# NVCC compiler options:
NVCC 		= nvcc
NVCC_FLAGS 	= -dc -arch compute_75
NVCC_LIBS 	=

# Linker options
LINKER 		 = nvcc #g++ #clang
LINKER_FLAGS = #-std=c++11 -O2 #-Wall -pedantic
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

$(TARGET): $(OBJS)
	$(LINKER) $(LINKER_FLAGS) $? -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

$(OBJ_DIR)/%.cu.o: $(SRC_DIR)/%.cu $(CUH_FILES)
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp $(H_FILES)
	$(CC) $(CC_FLAGS) -c $< -o $@

clean:
	$(RM) obj/* *.o $(TARGET)