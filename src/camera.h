#ifndef CAMERA_H
#define CAMERA_H

#include "vector.cuh"

struct Camera {
    Camera() = default;
    Camera(
            vec3 position, 
            vec3 direction, 
            ivec2 resolution
    ) {
        this->position       = position;
        this->direction      = direction;
        this->resolution     = resolution;
        this->focal_length   = 1.0f; // pretty wide - by default
        this->up             = vec3(0, 1, 0);
        this->aperture       = 0.0f; // pinhole style - by default
        this->focus_distance = 100.0f; // doesn't matter with pinhole style
    }
    vec3 position;
    vec3 direction;
    vec3 up;
    ivec2 resolution;
    float focal_length;
    float aperture;
    float focus_distance;
};

#endif