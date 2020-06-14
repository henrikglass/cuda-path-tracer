#ifndef MATERIAL_H
#define MATERIAL_H

#include "vector.cuh"

struct Material {
    Material() {}
    Material(vec3 albedo, float emission, float smoothness, bool smooth_shading) {
        this->albedo         = albedo;
        this->emission       = emission;
        this->smoothness     = smoothness;
        this->smooth_shading = smooth_shading;
    }
    vec3 albedo;
    float emission;
    float smoothness;
    bool smooth_shading;
};

#endif