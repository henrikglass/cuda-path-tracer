#ifndef RENDERER_H
#define RENDERER_H

#define PI 3.14159265f

#include <vector>

#include "camera.h"
#include "scene.cuh"
#include "vector.cuh"
#include "image.h"

Image render(const Camera &camera, Scene &scene);

__device__ float crand(float &seed);
__device__ vec3 sample_hemisphere(vec3 normal, float &seed);
__device__ mat3 get_tangent_space(vec3 normal);

__global__
void device_render(vec3 *buf, int buf_size, Camera camera, Entity *entities, int n_entities);

#endif