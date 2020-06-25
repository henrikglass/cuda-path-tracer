#include <stdio.h>
#include <iostream>

#include "renderer.cuh"
#include "geometry.cuh"
#include "util.cuh"
#include "vector.cuh"

void normalize_and_gamma_correct(
        std::vector<vec3> &buf, 
        int n_samples_per_pixel, 
        float gamma
) {
    for (int i = 0; i < buf.size(); i++) {
        buf[i] = buf[i] / n_samples_per_pixel;
        buf[i].x = pow(buf[i].x, 1.0f / gamma);
        buf[i].y = pow(buf[i].y, 1.0f / gamma);
        buf[i].z = pow(buf[i].z, 1.0f / gamma);
    }
}

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
    int tile_size = 16; // 16x16 pixels
    int samples_per_pixel = 1;
    int n_samples_total = n_pixels * samples_per_pixel;
    dim3 blocks(
            camera.resolution.x / tile_size + 1, 
            camera.resolution.y / tile_size + 1
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
    gpuErrchk(cudaMalloc(&d_rand_state, n_pixels*sizeof(curandState)));
    render_init<<<blocks, threads>>>(camera, d_rand_state);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // add z dimension for # of samples per pixel
    blocks.z = samples_per_pixel;

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

    // normalize and gamma correct image
    normalize_and_gamma_correct(result_pixels, 1000, 2.2f);

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
    curand_init(clock(), pixel_idx, 0, &rand_state[pixel_idx]);
}

__global__
void device_render(vec3 *buf, int buf_size, Camera camera, Entity *entities, int n_entities, curandState *rand_state) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_idx = y * camera.resolution.x +  x;
    if ((x >= camera.resolution.x) || (y >= camera.resolution.y))
        return;
    //if (x != 290 || y != 590)
    //    if (x != 160 || y != 400)
            //return;


    curandState &local_rand_state = rand_state[pixel_idx];
    
    
    //create ray
    vec3 ray_orig = camera.position;
    float n_x = (float(x) / float(camera.resolution.x)) - 0.5f;
    float n_y = (float(y) / float(camera.resolution.y)) - 0.5f;
    float aspect_ratio = float(camera.resolution.x) / float(camera.resolution.y);
    vec3 camera_right = -cross(camera.direction, camera.up);
    vec3 point = n_x * camera_right * aspect_ratio - n_y * camera.up +
                 camera.position + camera.direction*camera.focal_length;
    vec3 ray_dir = point - camera.position;
    /*vec3 ray_dir = vec3(
        x - (camera.resolution.x / 2),
        -y + (camera.resolution.y / 2),
        camera.focal_length
    );*/
    ray_dir.normalize();
    //vec3 ray_energy = vec3(1.0f, 1.0f, 1.0f);
    Ray ray(ray_orig, ray_dir/*, ray_energy*/);

    vec3 result(0, 0, 0);
    int ns = 1000; // 1000
    for (int i = 0; i < ns; i++) {
        ray.origin      = ray_orig;
        ray.direction   = ray_dir;
        //ray.energy      = ray_energy;
        result = result + color(ray, entities, n_entities, &local_rand_state);
    }
    //result = result / float(ns);
    //float gamma = 2.2;
    buf[pixel_idx].x += /*powf(*/result.x/*, 1.0f/gamma);*/;
    buf[pixel_idx].y += /*powf(*/result.y/*, 1.0f/gamma);*/;
    buf[pixel_idx].z += /*powf(*/result.z/*, 1.0f/gamma);*/;

    // color pixel
    //buf[pixelIdx].x += result.x;
    //buf[pixelIdx].y += result.y;
    //buf[pixelIdx].z += result.z;

    // ********** debug ************
    //Intersection hit;
    //if (!get_closest_intersection_in_scene(ray, entities, n_entities, hit))
    //    return;

    // normal
    //buf[pixelIdx] =  (hit.normal + vec3(1,1,1)) / 2;
    
    // albedo
    //buf[pixelIdx].x = hit.entity->material.albedo.x;
    //buf[pixelIdx].y = hit.entity->material.albedo.y;
    //buf[pixelIdx].z = hit.entity->material.albedo.z;

}

__device__ vec3 color(const Ray &ray, Entity *entities, int n_entities, curandState *local_rand_state) {
    vec3 attenuation(1.0f, 1.0f, 1.0f);
    vec3 result(0.0f, 0.0f, 0.0f);
    Ray cray = ray;
    for (int i = 0; i < 6; i++) {
        Intersection hit;
        if (get_closest_intersection_in_scene(cray, entities, n_entities, hit)) {
            Material m = hit.entity->material;
            //printf("\nhit\n");
            //printf("smoothness: %g\n", m.smoothness);
            //printf("color: (%g, %g, %g)\n", m.albedo.x, m.albedo.y, m.albedo.z);
            //printf("emmision: %g\n", m.emission);

            // eh?
            vec3 specular = m.specular;
            vec3 albedo = min(vec3(1.0f, 1.0f, 1.0f) - m.specular, m.albedo);
            float spec_chance = dot(specular, vec3(1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f));
            float diff_chance = dot(albedo, vec3(1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f));
            float sum = spec_chance + diff_chance;
            spec_chance /= sum;
            diff_chance /= sum;

            //printf("spec chance: %g\n", spec_chance);
            //printf("diff chance: %g\n", diff_chance);

            float roulette = curand_uniform(local_rand_state);
            if (roulette < spec_chance) {
                // specular reflection
                //printf("specular reflection: %g\n", diff_chance);
                float alpha   = powf(1000.0f, m.smoothness * m.smoothness);
                cray.origin    = hit.position + hit.normal * 0.001f;
                cray.direction = sample_hemisphere(reflect(cray.direction, hit.normal), alpha, local_rand_state);
                //cray.direction = reflect(cray.direction, hit.normal).normalized();
                float f       = (alpha + 2) / (alpha + 1);
                attenuation   = attenuation * (1.0f / spec_chance) * specular * f * dot(hit.normal, cray.direction);
            } else {
                // diffuse reflection
                //printf("diffuse reflection: %g\n", diff_chance);
                //result        = result + m.emission * albedo * attenuation;
                result        = result + m.emission * m.albedo * attenuation;
                cray.origin    = hit.position + hit.normal * 0.001f;
                //printf("position: (%g, %g, %g)\n", cray.origin.x, cray.origin.y, cray.origin.z);
                //printf("incoming cray dir: (%g, %g, %g)\n", cray.direction.x, cray.direction.y, cray.direction.z);
                cray.direction = sample_hemisphere(hit.normal, 1.0f, local_rand_state).normalized();
                //printf("outgoing cray dir: (%g, %g, %g)\n", cray.direction.x, cray.direction.y, cray.direction.z);
                attenuation   = attenuation * (1.0f / diff_chance) * albedo;
            }
            
            
            
            
            //result        = result + m.emission * m.albedo * attenuation;
            //ray.origin    = hit.position + hit.normal * 0.001f;
            //ray.direction = sample_hemisphere(hit.normal, local_rand_state);
            //attenuation   = attenuation * 2 * m.albedo * dot(hit.normal, ray.direction);
        } else {
            // fake ambient
            //printf("miss!\n");
            //result = result + attenuation * 0.03f * vec3(0.5f, 0.7f, 1.0f);
            break;
        }
    }
    return result;
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

__device__ vec3 reflect(const vec3 &dir, const vec3 &normal) {
    return dir - 2.0f * dot(dir, normal) * normal;
}

__device__ vec3 sample_hemisphere(const vec3 &dir, float alpha, curandState *local_rand_state) {
    //float x, y, z, d;
    //do {
    //    x = 2 * curand_uniform(local_rand_state) - 1;
    //    y = 2 * curand_uniform(local_rand_state) - 1;
    //    z = 2 * curand_uniform(local_rand_state) - 1;
    //    d = sqrtf(x*x + y*y + z*z);
    //} while(d > 1);
//
    //x= x / d;
    //y= y / d;
    //z= z / d;
    //vec3 v(x, y, z);
//
    //if (dot(dir, v) <= 0.0f)
    //    v = -v;
//
    //return v.normalized();
    
    float cos_theta = powf(curand_uniform(local_rand_state), 1.0f / (alpha + 1.0f));
    //float sin_theta = sqrtf(max(0.0f, 1.0f - cos_theta * cos_theta));
    float sin_theta = sqrtf(1.0f - cos_theta * cos_theta);
    float phi = 2 * PI * curand_uniform(local_rand_state);
    vec3 tangent_space_dir = vec3(cosf(phi) * sin_theta, sinf(phi) * sin_theta, cos_theta);
    // Transform direction to world space
    return tangent_space_dir * get_tangent_space(dir);
}

__device__ mat3 get_tangent_space(const vec3 &normal) {
    // Choose a helper vector for the cross product
    vec3 helper = vec3(1, 0, 0);
    if (fabsf(normal.x) > 0.99f)
        helper = vec3(0, 0, 1);
    // Generate vectors
    vec3 tangent = cross(normal, helper).normalized();
    vec3 binormal = cross(normal, tangent).normalized();
    return mat3(tangent, binormal, normal);
}