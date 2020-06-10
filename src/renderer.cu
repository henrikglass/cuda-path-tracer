#include <stdio.h>
#include <iostream>

#include "renderer.cuh"
#include "camera.cuh"
#include "geometry.cuh"
#include "util.cuh"
#include "vector.cuh"


void render(const Camera &camera, Scene &scene) {
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
    int buf_size = camera.resolution.x * camera.resolution.y * sizeof(vec3);
    vec3 *buf;
    gpuErrchk(cudaMalloc(&buf, buf_size));

    // move scene to device memory
    scene.copyToDevice();

    // render on device
    //ladug<<<1, 1>>>();
    device_render<<<1, 1>>>(buf, buf_size, camera, scene.d_entities, scene.entities.size());
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
}

__global__
void device_render(vec3 *buf, int buf_size, const Camera& camera, Entity *entities, int n_entities) {
    printf("%d\n", n_entities);
    for(int i = 0; i < n_entities; i++) {
        printf("entity %d has shape %d\n", i, entities[i].shape);
        printf("xyzr: %g %g %g %g\n", 
                entities[i].center.x, 
                entities[i].center.y, 
                entities[i].center.z, 
                entities[i].center.x
        );
    }
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