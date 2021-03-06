#include <stdio.h>
#include <iostream>

#include "renderer.cuh"
#include "geometry.cuh"
#include "util.cuh"
#include "vector.cuh"
#include "post_processing.cuh"

// definitions put in header because having them
// in multiple separate compilation units impacts
// performance negatively.
#include "device_geometry_functions.cuh"

/**
 * Initializes a curandState object for each pixel in the scene and
 * stores it in `rand_state`.
 */
__global__
void render_init(Camera camera, curandState *rand_state) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_idx = y * camera.resolution.x +  x;
    if ((x >= camera.resolution.x) || (y >= camera.resolution.y))
        return;
    curand_init(1337, pixel_idx, 0, &rand_state[pixel_idx]);
}

/**
 * Renders an image on the device, given a render configuration `config`.
 */
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

/**
 * Create a ray originating from the camera.
 *
 * @param camera            a camera object
 * @param u, v              the pixel coordinate
 * @param local_rand_state  a curandState object
 */
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

/**
 * Compute a single pixel color sample.
 * 
 * @param ray               a ray originating from the camera
 * @param scene             the scene representation
 * @param local_rand_state  a curandState object
 */
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

/**
 * Perfectly reflect the `dir` given a normal vector of the surface `normal`
 */
__device__ vec3 reflect(vec3 dir, vec3 normal) {
    return dir - 2.0f * dot(dir, normal) * normal;
}

/**
 * Samples a hemisphere around `dir`.
 *
 * @param dir       vector describing the orientation of the hemisphere to sample
 * @param alpha     the phong alpha, obtained from a materials `smoothness`
 */
__device__ vec3 sample_hemisphere(vec3 dir, float alpha, curandState *local_rand_state) {    
    float cos_theta = __powf(curand_uniform(local_rand_state), 1.0f / (alpha + 1.0f));
    float sin_theta = __fsqrt_rn(1.0f - cos_theta * cos_theta);
    float phi = 2 * PI * curand_uniform(local_rand_state);
    vec3 tangent_space_dir = vec3(__cosf(phi) * sin_theta, __sinf(phi) * sin_theta, cos_theta);
    return tangent_space_dir * get_tangent_space(dir);
}

/**
 * Construct a tangent space basis given a normal vector.
 */
__device__ mat3 get_tangent_space(vec3 normal) {
    vec3 helper = vec3(1, 0, 0);
    if (fabsf(normal.x) > 0.99f)
        helper = vec3(0, 0, 1);
    vec3 tangent = cross(normal, helper).normalized();
    vec3 binormal = cross(normal, tangent).normalized();
    return mat3(tangent, binormal, normal);
}

/**
 * Helper function for calculating `n_samples_per_pass` given a number of samples
 * per pixel `spp`.
 */
void Renderer::set_samples_per_pixel(unsigned int spp) {
    this->n_samples_per_pass = max(spp / (this->n_blocks_per_tile * this->n_split_buffers), 1);
}

/**
 * Wrapper function for rendering a scene and applying some post processing on 
 * the device.
 *
 * @param camera    a camera object
 * @param scene     a scene representation
 */
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
    compound_buffers(result_pixels, h_buf, this->n_split_buffers);

    // normalize and gamma correct image
    //tonemap(result_pixels, 32, 2.2f);
    int n_samples_per_pixel = this->n_split_buffers * 
            this->n_blocks_per_tile * this->n_samples_per_pass;
    
    // **********************************************
    // *************** post process *****************
    // **********************************************
    
    normalize_image(result_pixels, n_samples_per_pixel);
    
    // PP on HDR image
    std::vector<vec3> bright_parts = apply_threshold(result_pixels, 3.0f);
    ////result_pixels = apply_threshold(result_pixels, 1.0f);
    Kernel k;
    ////k.make_mean(2); // 5x5 kernel
    k.make_gaussian(32, 10.0f);
    //k.print();
    ////exit(0);
    apply_filter(bright_parts, camera.resolution, k);
    image_add(result_pixels,  bright_parts, 0.5f);

    Kernel kk;
    kk.make_gaussian(2, 0.7f);
    //apply_filter(result_pixels, camera.resolution, kk, BILATERAL);

    apply_aces(result_pixels);
    //apply_filter(result_pixels, camera.resolution, k);
    gamma_correct(result_pixels, 2.2f);
    apply_filter(result_pixels, camera.resolution, kk, BILATERAL);
    
    // **********************************************
    // **********************************************
    // **********************************************

    // free scene from device memory (should not be necessary, but why not)
    std::cout << "freeing scene from device..." << std::endl;
    gpuErrchk(cudaFree(d_scene));
    scene.free_from_device();
    std::cout << "done!" << std::endl;

    // return result
    // @Incomplete MUST cudaFree(buf);
    return Image(result_pixels, camera.resolution);
}
