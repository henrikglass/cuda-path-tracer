#ifndef CAMERA_H
#define CAMERA_H

#include "vector.cuh"

struct Camera {
    Camera(vec3 position, vec3 direction, vec2 resolution) {
        this->position   = position;
        this->direction  = direction;
        this->resolution = resolution;
    }
    vec3 position;
    vec3 direction;
    vec2 resolution;
}

#endif