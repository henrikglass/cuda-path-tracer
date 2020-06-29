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
    Scene *d_scene;
    gpuErrchk(cudaMalloc(&d_scene, sizeof(Scene)));
    cudaMemcpy(d_scene, &scene, sizeof(Scene), cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError());

    // device info debug print
    int devID = 0;
    cudaDeviceProp props;

    //Get GPU information
    cudaGetDevice(&devID);
    cudaGetDeviceProperties(&props, devID);
    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           devID, props.name, props.major, props.minor);

    // Decide on tile size, # of threads and # of blocks
    int tile_size = 8; // 16x16 pixels
    //int samples_per_pixel = 10;
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
    //blocks.z = samples_per_pixel;

    // setup RenderConfig
    RenderConfig config;
    config.buf        = buf;
    config.buf_size   = buf_size;
    config.camera     = camera;
    config.scene      = d_scene;
    config.rand_state = d_rand_state;
    config.n_samples  = 2000;

    std::cout << "start render" << std::endl;
    // render on device
    device_render<<<blocks, threads>>>(config);
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());

    // copy data back to host
    std::vector<vec3> result_pixels(n_pixels);
    cudaMemcpy(&(result_pixels[0]), buf, buf_size, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());

    // normalize and gamma correct image
    normalize_and_gamma_correct(result_pixels, config.n_samples, 2.2f);

    // free scene from device memory (should not be necessary, but why not)
    std::cout << "freeing scene from device..." << std::endl;
    gpuErrchk(cudaFree(d_scene));
    scene.free_from_device();
    std::cout << "done!" << std::endl;

    // return result
    return Image(result_pixels, camera.resolution);
}

__global__
void render_init(Camera camera, curandState *rand_state) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_idx = y * camera.resolution.x +  x;
    if ((x >= camera.resolution.x) || (y >= camera.resolution.y))
        return;
    curand_init(1337, pixel_idx, 0, &rand_state[pixel_idx]);
}

__global__
void device_render(RenderConfig config) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_idx = v * config.camera.resolution.x +  u;
    if ((u >= config.camera.resolution.x) || (v >= config.camera.resolution.y))
        return;
    
    //if (u != 10 || v != 10)
    //    return;

    //Intersection hit;
    //Ray ray = create_ray(config.camera, u, v);
    //if (!get_closest_intersection_in_scene(
    //        ray, 
    //        config.scene->d_entities, 
    //        config.scene->n_entities, 
    //        hit
    //)) {
    //    return;
    //}
//
    //// normal
    //config.buf[pixel_idx] =  (hit.normal + vec3(1,1,1)) / 2;

    curandState &local_rand_state = config.rand_state[pixel_idx];

    // color pixel
    vec3 result(0, 0, 0);
    int ns = config.n_samples; // 1000
    for (int i = 0; i < ns; i++) {
        Ray ray = create_ray(config.camera, u, v, &local_rand_state);
        result = result + color(ray, config.scene, &local_rand_state);
    }

    config.buf[pixel_idx] = config.buf[pixel_idx] + result;

}

__device__ Ray create_ray(Camera camera, int u, int v, curandState *local_rand_state) {
    vec3 ray_orig = camera.position;
    float n_u = (float(u + curand_uniform(local_rand_state)) / float(camera.resolution.x)) - 0.5f;
    float n_v = (float(v + curand_uniform(local_rand_state)) / float(camera.resolution.y)) - 0.5f;
    float aspect_ratio = float(camera.resolution.x) / float(camera.resolution.y);
    vec3 camera_right = -cross(camera.direction, camera.up);
    vec3 point = n_u * camera_right * aspect_ratio - n_v * camera.up +
                 camera.position + camera.direction*camera.focal_length;
    vec3 ray_dir = point - camera.position;
    ray_dir.normalize();
    return Ray(ray_orig, ray_dir);
}

__device__ vec3 color(Ray &ray, Scene *scene, curandState *local_rand_state) {
    vec3 attenuation(1.0f, 1.0f, 1.0f);
    vec3 result(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 6; i++) {
        Intersection hit;
        if (get_closest_intersection_in_scene(ray, scene->d_entities, scene->n_entities, hit)) {
            Material m = hit.entity->material;
            vec3 specular = m.specular;
            vec3 albedo = min(vec3(1.0f, 1.0f, 1.0f) - m.specular, m.albedo);
            float spec_chance = dot(specular, vec3(1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f));
            float diff_chance = dot(albedo, vec3(1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f));
            float sum = spec_chance + diff_chance;
            spec_chance /= sum;
            diff_chance /= sum;

            float roulette = curand_uniform(local_rand_state);
            if (roulette < spec_chance) {
                // specular reflection
                float alpha   = __powf(1000.0f, m.smoothness * m.smoothness);
                ray.origin    = hit.position + hit.normal * 0.001f;
                ray.direction = sample_hemisphere(reflect(ray.direction, hit.normal), alpha, local_rand_state);
                ray.recalc_fracs();
                float f       = (alpha + 2) / (alpha + 1);
                attenuation   = attenuation * (1.0f / spec_chance) * specular * f * dot(hit.normal, ray.direction);
            } else {
                // diffuse reflection
                result        = result + m.emission * m.albedo * attenuation;
                ray.origin    = hit.position + hit.normal * 0.001f;
                ray.direction = sample_hemisphere(hit.normal, 1.0f, local_rand_state);
                ray.recalc_fracs();
                attenuation   = attenuation * (1.0f / diff_chance) * albedo;
            }
        } else {
            // sample environment
            result = result + attenuation * scene->sample_hdri(ray.direction);
            break;
        }
    }
    return result;
}

__device__ vec3 reflect(const vec3 &dir, const vec3 &normal) {
    return dir - 2.0f * dot(dir, normal) * normal;
}

__device__ vec3 sample_hemisphere(const vec3 &dir, float alpha, curandState *local_rand_state) {    
    float cos_theta = __powf(curand_uniform(local_rand_state), 1.0f / (alpha + 1.0f));
    float sin_theta = __fsqrt_rn(1.0f - cos_theta * cos_theta);
    float phi = 2 * PI * curand_uniform(local_rand_state);
    vec3 tangent_space_dir = vec3(__cosf(phi) * sin_theta, __sinf(phi) * sin_theta, cos_theta);
    return tangent_space_dir * get_tangent_space(dir);
}

__device__ mat3 get_tangent_space(const vec3 &normal) {
    vec3 helper = vec3(1, 0, 0);
    if (fabsf(normal.x) > 0.99f)
        helper = vec3(0, 0, 1);
    vec3 tangent = cross(normal, helper).normalized();
    vec3 binormal = cross(normal, tangent).normalized();
    return mat3(tangent, binormal, normal);
}

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