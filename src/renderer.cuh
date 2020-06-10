#ifndef RENDERER_H
#define RENDERER_H
#include "camera.cuh"
#include "scene.cuh"

void render(const Camera &camera, Scene &scene);

__global__
void device_render(vec3 *buf, int buf_size, const Camera& camera, Entity *entities, int n_entities);

#endif