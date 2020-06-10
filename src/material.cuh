#ifndef MATERIAL_H
#define MATERIAL_H

#include "vector.cuh"

struct Material {
    Material() {}
    Material(vec3 albedo, float emission, float smoothness) {
        this->albedo     = albedo;
        this->emission   = emission;
        this->smoothness = smoothness;
    }
    vec3 albedo;
    float emission;
    float smoothness;
};

#endif