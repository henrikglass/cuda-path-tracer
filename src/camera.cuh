#ifndef CAMERA_H
#define CAMERA_H

#include "vector.cuh"

struct Camera {
    Camera(vec3 position, vec3 direction, vec2 resolution, float focal_length) {
        this->position      = position;
        this->direction     = direction;
        this->resolution    = resolution;
        this->focal_length  = focal_length;
    }
    vec3 position;
    vec3 direction;
    vec2 resolution;
    float focal_length;
};

#endif