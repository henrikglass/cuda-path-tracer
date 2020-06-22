#include <stdio.h>
#include <iostream>

#include "renderer.cuh"
#include "geometry.cuh"
#include "util.cuh"
#include "vector.cuh"


Image render(const Camera &camera, Scene &scene) {
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
    int samples_per_pixel = 1;
    int n_samples_total = n_pixels * samples_per_pixel;
    dim3 blocks(
            camera.resolution.x / tile_size + 1, 
            camera.resolution.y / tile_size + 1/*,
            samples_per_pixel*/
    );
    dim3 threads(tile_size, tile_size);

    // set stack size limit. (Default proved too little for deep octrees)
    size_t limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    size_t new_limit = 1024 << 5;
    cudaDeviceSetLimit( cudaLimitStackSize, new_limit );
    std::cout << "device stack limit: " << new_limit << "KiB" << std::endl;

    // curand setup
    curandState *d_rand_state;
    gpuErrchk(cudaMalloc(&d_rand_state, n_samples_total*sizeof(curandState)));
    render_init<<<blocks, threads>>>(camera, d_rand_state);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    std::cout << "start render" << std::endl;
    // render on device
    device_render<<<blocks, threads>>>(buf, buf_size, camera, scene.d_entities, scene.entities.size(), d_rand_state);
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
void render_init(Camera camera, curandState *rand_state) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_idx = y * camera.resolution.x +  x;
    if ((x >= camera.resolution.x) || (y >= camera.resolution.y))
        return;
    curand_init(1984, pixel_idx, 0, &rand_state[pixel_idx]);
}

__global__
void device_render(vec3 *buf, int buf_size, Camera camera, Entity *entities, int n_entities, curandState *rand_state) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_idx = y * camera.resolution.x +  x;
    if ((x >= camera.resolution.x) || (y >= camera.resolution.y))
        return;
    //if (x != 496 || y != 424)
    //    return;


    curandState local_rand_state = rand_state[pixel_idx];
    
    
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

    vec3 result(0, 0, 0);
    int ns = 1000;
    for (int i = 0; i < ns; i++) {
        ray.origin      = ray_orig;
        ray.direction   = ray_dir;
        ray.energy      = ray_energy;
        result = result + color(ray, entities, n_entities, &local_rand_state);
    }
    result = result / float(ns);
    buf[pixel_idx].x += sqrtf(result.x);
    buf[pixel_idx].y += sqrtf(result.y);
    buf[pixel_idx].z += sqrtf(result.z);

    // color pixel
    //buf[pixelIdx].x += result.x;
    //buf[pixelIdx].y += result.y;
    //buf[pixelIdx].z += result.z;

    // normal
    //buf[pixelIdx] = vec3(1.0f, 0.0f, 1.0f);
    //buf[pixelIdx] =  (hit.normal + vec3(1,1,1)) / 2;
    
    // albedo
    //buf[pixelIdx].x = hit.entity->material.albedo.x;
    //buf[pixelIdx].y = hit.entity->material.albedo.y;
    //buf[pixelIdx].z = hit.entity->material.albedo.z;

}

__device__ vec3 color(Ray &ray, Entity *entities, int n_entities, curandState *local_rand_state) {
    Intersection hit;
    //float cur_attenuation = 1.0f;
    for (int i = 0; i < 50; i++) {
        if (get_closest_intersection_in_scene(ray, entities, n_entities, hit)) {
            Material &m = hit.entity->material;
            if (m.emission > 0.01f) {
                return ray.energy * m.emission;
            } else {
                ray.origin       = hit.position + hit.normal * 0.001f;
                ray.direction    = sample_hemisphere(hit.normal, local_rand_state);
                ray.energy       = (ray.energy * m.albedo) * saturate(dot(hit.normal, ray.direction));
                //cur_attenuation *= 0.5f;
            }

        } else {
            return vec3(0, 0, 0);
            //ray.direction.normalize();
            //float t = 0.5f*(ray.direction.y + 1.0f);
            //vec3 c = (1.0f - t) * vec3(1, 1, 1) + t * vec3(0.5, 0.7, 1.0);
            //return cur_attenuation * c;
        }
    }
    return vec3(0, 0, 0);
}

//__device__ float crand(float &seed) {
//    float garbage;
//    vec2 pixel(
//        blockIdx.x * blockDim.x + threadIdx.x, 
//        blockIdx.y * blockDim.y + threadIdx.y
//    );
//    float result = modff(sinf(seed / 100.0f * dot(pixel, vec2(12.9898f, 78.233f))) * 43758.5453f, &garbage);
//    seed += 1.0f;
//    return result;
//}

__device__ vec3 sample_hemisphere(vec3 normal, curandState *local_rand_state) {
    float cos_theta = curand_uniform(local_rand_state);
    float sin_theta = sqrtf(max(0.0f, 1.0f - cos_theta * cos_theta));
    float phi = 2 * PI * curand_uniform(local_rand_state);
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