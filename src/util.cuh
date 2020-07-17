#ifndef UTIL_H
#define UTIL_H

#include <stdio.h>
#include <vector>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    } 
}

template <typename T>
T* prepare_empty_cuda_buffer(size_t size) {
    T *d_buf;
    gpuErrchk(cudaMalloc(&d_buf, size * sizeof(T)));
    return d_buf;
}

template <typename T>
T* prepare_cuda_buffer(const std::vector<T> to_copy) {
    T *d_buf;
    size_t buf_size = to_copy.size() * sizeof(T);
    gpuErrchk(cudaMalloc(&d_buf, buf_size));
    gpuErrchk(cudaMemcpy(d_buf, &(to_copy[0]), buf_size, cudaMemcpyHostToDevice));
    return d_buf;
}

template <typename T>
T* prepare_cuda_instance(const T &to_copy) {
    T *d_instance;
    gpuErrchk(cudaMalloc(&d_instance, sizeof(T)));
    gpuErrchk(cudaMemcpy(d_instance, &(to_copy), sizeof(T), cudaMemcpyHostToDevice));
    return d_instance;
}

// not really math @Incomplete Move this somewhere more suitable
__host__ __device__ inline float luminosity(vec3 color) {
    return dot(color, vec3(0.21f, 0.72f, 0.07f));
}

#endif
