#ifndef RENDERER_H
#define RENDERER_H

#include <vector>

#include "camera.cuh"
#include "scene.cuh"
#include "vector.cuh"
#include "image.cuh"

Image render(const Camera &camera, Scene &scene);

__global__
void device_render(vec3 *buf, int buf_size, Camera camera, Entity *entities, int n_entities);

#endif