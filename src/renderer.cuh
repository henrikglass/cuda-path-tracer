#ifndef RENDERER_H
#define RENDERER_H

#define PI 3.14159265f

#include <vector>
#include <curand_kernel.h>

#include "camera.h"
#include "scene.cuh"
#include "vector.cuh"
#include "image.h"

Image render(const Camera &camera, Scene &scene);

__device__ vec3 sample_hemisphere(vec3 normal, curandState *local_rand_state);
__device__ mat3 get_tangent_space(vec3 normal);
__device__ vec3 color(Ray &ray, Entity *entities, int n_entities, curandState *local_rand_state);

__global__
void render_init(Camera camera, curandState *rand_state);

__global__
void device_render(
        vec3 *buf, 
        int buf_size, 
        Camera camera, 
        Entity *entities, 
        int n_entities, 
        curandState *rand_state
);

#endif