#include <stdio.h>
#include <iostream>

#include "renderer.cuh"
#include "camera.cuh"
#include "geometry.cuh"
#include "util.cuh"
#include "vector.cuh"


Image render(const Camera &camera, Scene &scene) {
    // ------------------------------- debug ------------------------------
    // debug print scene representation
    std::cout << "camera position: " << camera.position << " camera direction: " << camera.direction << std::endl;
    for (Entity e : scene.entities) {
        switch (e.shape) {
            case SPHERE:
                std::cout << "Sphere with radius " << e.radius << " at " << e.center << std::endl;
                break;
            case TRIANGLE_MESH:
                std::cout << "triangle mesh" << std::endl;
                break;
            default:
                std::cout << "undefined entity" << std::endl;
                break;
        }
    }

    // device info debug print
    int devID = 0;
    cudaDeviceProp props;

    //Get GPU information
    cudaGetDevice(&devID);
    cudaGetDeviceProperties(&props, devID);
    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           devID, props.name, props.major, props.minor);
    // ------------------------------- debug ------------------------------




    // Allocate output image buffer on device
    int n_pixels = camera.resolution.x * camera.resolution.y;
    int buf_size = n_pixels * sizeof(vec3);
    vec3 *buf;
    gpuErrchk(cudaMalloc(&buf, buf_size));

    // move scene to device memory
    scene.copyToDevice();

    // Decide on tile size, # of threads and # of blocks
    int tile_size = 8; // 8x8 pixels
    dim3 blocks(
            camera.resolution.x / tile_size + 1, 
            camera.resolution.y / tile_size + 1
    );
    dim3 threads(tile_size, tile_size);

    // render on device
    device_render<<<blocks, threads>>>(buf, buf_size, camera, scene.d_entities, scene.entities.size());
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    // copy data back to host
    std::vector<vec3> result_pixels(n_pixels);
    cudaMemcpy(&(result_pixels[0]), buf, buf_size, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());

    // return result
    return Image(result_pixels, camera.resolution);
}

__global__
void device_render(vec3 *buf, int buf_size, Camera camera, Entity *entities, int n_entities) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if ((i >= camera.resolution.x) || (j >= camera.resolution.y))
        return;
    int pixelIdx = j * camera.resolution.x +  i;
    buf[pixelIdx].x = float(i) / camera.resolution.x;
    buf[pixelIdx].y = float(j) / camera.resolution.y;
    buf[pixelIdx].z = 0.2f;
}

/*vec3 render(const vec3& a, const vec3& b) {
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
}*/