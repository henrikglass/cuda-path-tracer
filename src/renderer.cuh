#ifndef RENDERER_H
#define RENDERER_H

#define PI 3.14159265f

#include <vector>
#include <curand_kernel.h>

#include "camera.h"
#include "scene.cuh"
#include "vector.cuh"
#include "image.h"

struct RenderConfig {
    vec3 *buf;
    size_t buf_size;
    Camera camera;
    Scene *scene;
    curandState *rand_state;
    size_t n_samples;
};

// host
class Renderer {
private:
    void compound_buffers(std::vector<vec3> &out_image, const std::vector<vec3> &in_buf);
    void tonemap(std::vector<vec3> &buf, float gamma);
public:
    unsigned int n_split_buffers    = 4;
    unsigned int n_blocks_per_tile  = 8;
    unsigned int n_samples_per_pass = 32;
    unsigned int tile_size          = 4;
    void set_samples_per_pixel(unsigned int spp);
    Image render(const Camera &camera, Scene &scene);
};

Image render(const Camera &camera, Scene &scene);
void compound(std::vector<vec3> &out_image, const std::vector<vec3> &in_buf, int n_split_buffers);
void tonemap(std::vector<vec3> &buf, int n_samples_per_pixel, float gamma);

// device
__device__ vec3 reflect(vec3 dir, vec3 normal);
__device__ vec3 sample_hemisphere(vec3 dir, float alpha, curandState *local_rand_state);
__device__ mat3 get_tangent_space(vec3 normal);
__device__ vec3 color(Ray &ray, Scene *scene, curandState *local_rand_state);
__device__ Ray create_camera_ray(Camera camera, int u, int v, curandState *local_rand_state);

// kernels
__global__
void render_init(Camera camera, curandState *rand_state);

__global__
void device_render(RenderConfig config);

#endif
