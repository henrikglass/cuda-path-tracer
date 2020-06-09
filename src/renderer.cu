#include "renderer.cuh"
#include "util.cuh"
#include "vector.cuh"
#include <stdio.h>


vec3 render(const vec3& a, const vec3& b) {
    int devID = 0;
    cudaDeviceProp props;

    //Get GPU information
    cudaGetDevice(&devID);
    cudaGetDeviceProperties(&props, devID);
    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           devID, props.name, props.major, props.minor);

    vec3 ans;
    vec3 *d_ans;
    cudaMalloc(&d_ans, sizeof(vec3));
    // call ladug
    ladug<<<1, 2>>>(d_ans, a, b);
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(&ans, d_ans, sizeof(vec3), cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    return ans;
}

__global__
void ladug(vec3 *ans, const vec3 a, const vec3 b) {
    *ans = a + b;
    printf("hello\n");
}