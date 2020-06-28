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
Image render(const Camera &camera, Scene &scene);
void normalize_and_gamma_correct(std::vector<vec3> &buf, int n_samples_per_pixel, float gamma);

// device
__device__ vec3 reflect(const vec3 &dir, const vec3 &normal);
__device__ vec3 sample_hemisphere(const vec3 &dir, float alpha, curandState *local_rand_state);
__device__ mat3 get_tangent_space(const vec3 &normal);
__device__ vec3 color(Ray &ray, Scene *scene, curandState *local_rand_state);
__device__ Ray create_ray(Camera camera, int u, int v);

// kernels
__global__
void render_init(Camera camera, curandState *rand_state);

__global__
void device_render(RenderConfig config);

#endif