#include <stdio.h>
#include <iostream>

#include "renderer.cuh"
#include "geometry.cuh"
#include "util.cuh"
#include "vector.cuh"


Image render(const Camera &camera, Scene &scene) {
    // ------------------------------- debug ------------------------------
    // debug print scene representation
    std::cout << "camera position: " << camera.position << " camera direction: " << camera.direction << std::endl;
    for (Entity *e : scene.entities) {
        switch (e->shape) {
            case SPHERE:
                std::cout << "Sphere" << std::endl;
                break;
            case TRIANGLE_MESH:
                std::cout << "triangle mesh" << std::endl;
                break;
            default:
                std::cout << "undefined entity" << std::endl;
                break;
        }
    }

    // ------------------------------- debug ------------------------------




    // Allocate output image buffer on device
    int n_pixels = camera.resolution.x * camera.resolution.y;
    int buf_size = n_pixels * sizeof(vec3);
    vec3 *buf;
    gpuErrchk(cudaMalloc(&buf, buf_size));

    // move scene to device memory
    std::cout << "copying scene to device..." << std::endl;
    scene.copy_to_device();
    std::cout << "done!" << std::endl;

    // device info debug print
    int devID = 0;
    cudaDeviceProp props;

    //Get GPU information
    cudaGetDevice(&devID);
    cudaGetDeviceProperties(&props, devID);
    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           devID, props.name, props.major, props.minor);

    // Decide on tile size, # of threads and # of blocks
    int tile_size = 8; // 8x8 pixels
    int samples_per_pixel = 1000;
    dim3 blocks(
            camera.resolution.x / tile_size + 1, 
            camera.resolution.y / tile_size + 1,
            samples_per_pixel
    );
    dim3 threads(tile_size, tile_size);

    // set stack size limit. (Default proved too little for deep octrees)
    size_t limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    size_t new_limit = 1024 << 5;
    cudaDeviceSetLimit( cudaLimitStackSize, new_limit );
    std::cout << "device stack limit: " << new_limit << "KiB" << std::endl;

    // render on device
    device_render<<<blocks, threads>>>(buf, buf_size, camera, scene.d_entities, scene.entities.size());
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    // copy data back to host
    std::vector<vec3> result_pixels(n_pixels);
    cudaMemcpy(&(result_pixels[0]), buf, buf_size, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());

    // free scene from device memory (should not be necessary, but why not)
    std::cout << "freeing scene from device..." << std::endl;
    scene.free_from_device();
    std::cout << "done!" << std::endl;

    // return result
    //std::vector<vec3> result_pixels(n_pixels);
    return Image(result_pixels, camera.resolution);
}


__global__
void device_render(vec3 *buf, int buf_size, Camera camera, Entity *entities, int n_entities) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ((x >= camera.resolution.x) || (y >= camera.resolution.y))
        return;
    //if (x != 496 || y != 424)
    //    return;



    int pixelIdx = y * camera.resolution.x +  x;
    
    //create ray
    vec3 ray_orig = camera.position;
    vec3 ray_dir = vec3(
        x - (camera.resolution.x / 2),
        -y + (camera.resolution.y / 2),
        camera.focal_length
    );
    ray_dir.normalize();
    vec3 ray_energy = vec3(1.0f, 1.0f, 1.0f);
    Ray ray(ray_orig, ray_dir, ray_energy);

    // cast ray
    vec3 result(0, 0, 0);
    float seed = pixelIdx*blockIdx.z / 913.315f;
    for (int i = 0; i < 3; i++) {
        Intersection hit;
        if(!get_closest_intersection_in_scene(ray, entities, n_entities, hit))
            break; // no hit
        
        if (hit.entity->material.emission > 0.001f) {
            result.x = hit.entity->material.emission * hit.entity->material.albedo.x * ray.energy.x;
            result.y = hit.entity->material.emission * hit.entity->material.albedo.y * ray.energy.y;
            result.z = hit.entity->material.emission * hit.entity->material.albedo.z * ray.energy.z;
            break;
        }

        ray.origin = hit.position + hit.normal * 0.001f;
        ray.direction = sample_hemisphere(hit.normal, seed);
        ray.energy.x = ray.energy.x * 2 * hit.entity->material.albedo.x * saturate(dot(hit.normal, ray.direction));
        ray.energy.y = ray.energy.y * 2 * hit.entity->material.albedo.y * saturate(dot(hit.normal, ray.direction));
        ray.energy.z = ray.energy.z * 2 * hit.entity->material.albedo.z * saturate(dot(hit.normal, ray.direction));
    }

    // color pixel
    buf[pixelIdx].x += result.x;
    buf[pixelIdx].y += result.y;
    buf[pixelIdx].z += result.z;

    

    // normal
    //buf[pixelIdx] = vec3(1.0f, 0.0f, 1.0f);
    //buf[pixelIdx] =  (hit.normal + vec3(1,1,1)) / 2;
    
    // albedo
    //buf[pixelIdx].x = hit.entity->material.albedo.x;
    //buf[pixelIdx].y = hit.entity->material.albedo.y;
    //buf[pixelIdx].z = hit.entity->material.albedo.z;




    // simple gradient render
    //buf[pixelIdx].x = float(x) / camera.resolution.x;
    //buf[pixelIdx].y = float(y) / camera.resolution.y;
    //buf[pixelIdx].z = 0.2f;
}

__device__ float crand(float &seed) {
    float garbage;
    vec2 pixel(
        blockIdx.x * blockDim.x + threadIdx.x, 
        blockIdx.y * blockDim.y + threadIdx.y
    );
    float result = modff(sinf(seed / 100.0f * dot(pixel, vec2(12.9898f, 78.233f))) * 43758.5453f, &garbage);
    seed += 1.0f;
    return result;
}

__device__ vec3 sample_hemisphere(vec3 normal, float &seed) {
    float cos_theta = crand(seed);
    float sin_theta = sqrtf(max(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = 2 * PI * crand(seed);
    vec3 tangent_space_dir = vec3(cosf(phi) * sin_theta, sinf(phi) * sin_theta, cos_theta);
    // Transform direction to world space
    return tangent_space_dir * get_tangent_space(normal);
}

__device__ mat3 get_tangent_space(vec3 normal) {
    // Choose a helper vector for the cross product
    vec3 helper = vec3(1, 0, 0);
    if (fabsf(normal.x) > 0.99f)
        helper = vec3(0, 0, 1);
    // Generate vectors
    vec3 tangent = cross(normal, helper).normalized();
    vec3 binormal = cross(normal, tangent).normalized();
    return mat3(tangent, binormal, normal);
}