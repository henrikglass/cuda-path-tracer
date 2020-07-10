#include <stdio.h>
#include <iostream>

#include "renderer.cuh"
#include "geometry.cuh"
#include "util.cuh"
#include "vector.cuh"

/*
 * Device side
 */

// definitions put in header because having them
// in multiple separate compilation units impacts
// performance negatively.
#include "device_geometry_functions.cuh"

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
    int buf_offset = threadIdx.z * (config.camera.resolution.x * config.camera.resolution.y);
    if ((u >= config.camera.resolution.x) || (v >= config.camera.resolution.y))
        return;

    // color pixel
    vec3 result(0, 0, 0);
    curandState &local_rand_state = config.rand_state[pixel_idx];
    int ns = config.n_samples; // 1000
    for (int i = 0; i < ns; i++) {
        Ray ray = create_camera_ray(config.camera, u, v, &local_rand_state);
        result = result + color(ray, config.scene, &local_rand_state);
    }

    config.buf[pixel_idx + buf_offset] = config.buf[pixel_idx + buf_offset] + result;
}

__device__ Ray create_camera_ray(Camera camera, int u, int v, curandState *local_rand_state) {
    // create perfect (pinhole) ray
    vec3 ray_orig = camera.position;
    float n_u = (float(u + curand_uniform(local_rand_state)) / float(camera.resolution.x)) - 0.5f;
    float n_v = (float(v + curand_uniform(local_rand_state)) / float(camera.resolution.y)) - 0.5f;
    float aspect_ratio = float(camera.resolution.x) / float(camera.resolution.y);
    vec3 camera_right = -cross(camera.direction, camera.up);
    vec3 point = n_u * camera_right * aspect_ratio - n_v * camera.up +
                 camera.position + camera.direction*camera.focal_length;
    vec3 ray_dir = point - camera.position;
    ray_dir.normalize();

    if (camera.aperture > 0.01f) { // should not be needed but hey
        // set origin to random point on aperture, adjust ray direction accordingly
        float r = __fsqrt_rn(curand_uniform(local_rand_state)) * (camera.aperture / 2);
        float alpha = curand_uniform(local_rand_state) * 2*PI;
        float dx = __cosf(alpha) * r;
        float dy = __sinf(alpha) * r;
        vec3 orig_offset = dx * camera_right - dy * camera.up; // twist to camera orientation
        vec3 focal_point = ray_orig + camera.focus_distance * 
                (1.0f / dot(camera.direction, ray_dir)) * ray_dir;
        ray_orig = ray_orig + orig_offset;
        ray_dir = focal_point - ray_orig;
        ray_dir.normalize();
    }

    return Ray(ray_orig, ray_dir);
}

__device__ float luminosity(vec3 color) {
    return dot(color, vec3(0.21f, 0.72f, 0.07f));
}

__device__ vec3 color(Ray &ray, Scene *scene, curandState *local_rand_state) {
    vec3 attenuation(1.0f, 1.0f, 1.0f);
    vec3 result(0.0f, 0.0f, 0.0f);
    for (int i = 0; i < 6; i++) {

        Intersection hit;
        if (!trace(ray, scene->d_entities, scene->n_entities, hit)) {
            result = result + attenuation * scene->sample_hdri(ray.direction);
            break;
        }

        // sample surface material properties
        vec3 ts_normal   = hit.entity->d_material->sample_normal(hit.u, hit.v);
        vec3 albedo      = hit.entity->d_material->sample_albedo(hit.u, hit.v);
        float smoothness = hit.entity->d_material->sample_smoothness(hit.u, hit.v);
        vec3 specular    = hit.entity->d_material->sample_specular(hit.u, hit.v);
        float emission   = hit.entity->d_material->emission;
        vec3 normal      = ts_normal * get_tangent_space(hit.normal);

        // decide whether to do specular or diffuse reflection
        albedo = min(1.0f - specular, albedo);
        float spec_chance = luminosity(specular);
        float diff_chance = luminosity(albedo);
        float sum = spec_chance + diff_chance;
        spec_chance /= sum;
        diff_chance /= sum;
        float roulette = curand_uniform(local_rand_state);

        if (roulette < spec_chance) {
            // specular reflection
            float alpha   = __powf(1000.0f, smoothness * smoothness);
            ray.origin    = hit.position + normal * 0.001f;
            ray.direction = sample_hemisphere(reflect(ray.direction, normal), alpha, local_rand_state);
            ray.recalc_fracs();
            float f       = (alpha + 2) / (alpha + 1);
            attenuation   = attenuation * (1.0f / spec_chance) * specular * f * dot(normal, ray.direction);
        } else {
            // diffuse reflection
            result        = result + emission * albedo * attenuation;
            ray.origin    = hit.position + normal * 0.001f;
            ray.direction = sample_hemisphere(normal, 1.0f, local_rand_state);
            ray.recalc_fracs();
            attenuation   = attenuation * (1.0f / diff_chance) * albedo;
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

/*
 * Host side image processing
 */

vec3 ACESFilm(vec3 x)
{
    // From: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    vec3 res = (x*(a*x+b))/(x*(c*x+d)+e);
    res.x = max(0.0f, min(1.0f, res.x));
    res.y = max(0.0f, min(1.0f, res.y));
    res.z = max(0.0f, min(1.0f, res.z));
    return res;
}

void Renderer::tonemap(
        std::vector<vec3> &buf, 
        float gamma
) {
    int n_samples_per_pixel = this->n_split_buffers * 
            this->n_blocks_per_tile * this->n_samples_per_pass;
    for (size_t i = 0; i < buf.size(); i++) {
        buf[i] = buf[i] / n_samples_per_pixel;
        buf[i] = ACESFilm(buf[i]);
        buf[i].x = pow(buf[i].x, 1.0f / gamma);
        buf[i].y = pow(buf[i].y, 1.0f / gamma);
        buf[i].z = pow(buf[i].z, 1.0f / gamma);
    }
}

void Renderer::compound_buffers(std::vector<vec3> &out_image, const std::vector<vec3> &in_buf) {
    std::cout << "capacity: " << out_image.capacity() << std::endl;
    size_t single_buffer_size = out_image.capacity();
    for (size_t i = 0; i < single_buffer_size; i++) {
        vec3 sum(0);
        for(unsigned int j = 0; j < this->n_split_buffers; j++) {
            sum = sum + in_buf[i + j * single_buffer_size];
        }
        out_image[i] = sum;
    }
}

void Renderer::set_samples_per_pixel(unsigned int spp) {
    this->n_samples_per_pass = max(spp / (this->n_blocks_per_tile * this->n_split_buffers), 1);
}

Image Renderer::render(const Camera &camera, Scene &scene) {
    // Allocate output image buffer on device
    int n_pixels = camera.resolution.x * camera.resolution.y;
    int buf_size = this->n_split_buffers * n_pixels * sizeof(vec3);
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

    //int samples_per_pixel = 10;
    dim3 blocks(
            camera.resolution.x / this->tile_size, 
            camera.resolution.y / this->tile_size
    );
    dim3 threads(this->tile_size, this->tile_size);

    // set stack size limit. (Default proved too little for deep octrees)
    size_t limit = 0;
    cudaDeviceGetLimit(&limit, cudaLimitStackSize);
    size_t new_limit = 1024 << 4;
    cudaDeviceSetLimit( cudaLimitStackSize, new_limit );
    std::cout << "device stack limit: " << new_limit << "KiB" << std::endl;

    // curand setup
    curandState *d_rand_state;
    gpuErrchk(cudaMalloc(&d_rand_state, n_pixels*sizeof(curandState)));
    render_init<<<blocks, threads>>>(camera, d_rand_state);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // add z dimension for # of samples per pixel
    blocks.z = this->n_blocks_per_tile;
    threads.z = this->n_split_buffers;

    // setup RenderConfig
    RenderConfig config;
    config.buf        = buf;
    config.buf_size   = buf_size;
    config.camera     = camera;
    config.scene      = d_scene;
    config.rand_state = d_rand_state;
    config.n_samples  = this->n_samples_per_pass;
    
    std::cout << "start render" << std::endl;
    // render on device
    device_render<<<blocks, threads>>>(config);
    gpuErrchk(cudaPeekAtLastError());
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    
    // copy data back to host
    std::vector<vec3> result_pixels(n_pixels);
    std::vector<vec3> h_buf(4*n_pixels);
    cudaMemcpy(&(h_buf[0]), buf, buf_size, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());

    // compound split buffers into single image
    compound_buffers(result_pixels, h_buf);

    // normalize and gamma correct image
    //tonemap(result_pixels, 32, 2.2f);
    tonemap(result_pixels, 2.2f);

    // free scene from device memory (should not be necessary, but why not)
    std::cout << "freeing scene from device..." << std::endl;
    gpuErrchk(cudaFree(d_scene));
    scene.free_from_device();
    std::cout << "done!" << std::endl;

    // return result
    return Image(result_pixels, camera.resolution);
}