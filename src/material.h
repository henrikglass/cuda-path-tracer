#ifndef MATERIAL_H
#define MATERIAL_H

#include "vector.cuh"

struct Material {
    Material() {}
    Material(vec3 albedo, vec3 specular, float emission, float smoothness, bool smooth_shading) {
        this->albedo             = albedo;
        this->specular           = specular;
        this->emission           = emission;
        this->smoothness         = smoothness;
        this->smooth_shading     = smooth_shading;
    }
    __device__ vec3 sample_albedo(float u, float v) {
        if(!this->has_albedo_texture)
            return this->albedo;
        
        return this->albedo;
    }
    vec3 albedo = vec3(0.8f);
    vec3 specular = vec3(0.0f);
    float emission = 0.0f;
    float smoothness = 0.0f;
    bool smooth_shading = false;
    bool has_albedo_texture = false;
};

#endif