#ifndef RENDERER_H
#define RENDERER_H
#include "vector.cuh"

vec3 render(const vec3& a, const vec3& b);

__global__
void ladug(vec3 *ans, const vec3 a, const vec3 b);

#endif