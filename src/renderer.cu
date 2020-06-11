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
    scene.copy_to_device();

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
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= camera.resolution.x) || (y >= camera.resolution.y))
        return;
    int pixelIdx = y * camera.resolution.x +  x;
    
    //create ray
    vec3 ray_orig = camera.position;
    vec3 ray_dir = vec3(
        x - (camera.resolution.x / 2),
        -y + (camera.resolution.y / 2),
        camera.focal_length
    );
    ray_dir.normalize();
    Ray ray(ray_orig, ray_dir);

    // cast ray
    Intersection hit;
    if(!get_closest_intersection_in_scene(ray, entities, n_entities, hit))
        return; // no hit

    // color pixel
    buf[pixelIdx] =  (hit.normal + vec3(1,1,1)) / 2;
    //buf[pixelIdx].x = hit.entity->material.albedo.x;
    //buf[pixelIdx].y = hit.entity->material.albedo.y;
    //buf[pixelIdx].z = hit.entity->material.albedo.z;

    // simple gradient render
    //buf[pixelIdx].x = float(x) / camera.resolution.x;
    //buf[pixelIdx].y = float(y) / camera.resolution.y;
    //buf[pixelIdx].z = 0.2f;
}