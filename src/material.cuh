#ifndef MATERIAL_H
#define MATERIAL_H

#include "vector.cuh"
#include "util.cuh"
#include "texture.cuh"

/*
 * Material
 */
struct Material {
    Material() {}
    ~Material();
    Material(vec3 albedo, vec3 specular, float emission, float smoothness, bool smooth_shading);
    
    // memory management
    __host__ void copy_to_device();
    __host__ void free_from_device();

    // device functions
    __device__ vec3 sample_albedo(float u, float v) {
        if (!albedo_map.is_set()) 
            return this->albedo;
        return albedo_map.sample(u, v);
    }

    __device__ vec3 sample_specular(float u, float v) {
        if (!specular_map.is_set()) 
            return this->specular;
        return specular_map.sample(u, v);
    }

    __device__ float sample_smoothness(float u, float v) {
        if (!smoothness_map.is_set()) 
            return this->smoothness;
        return smoothness_map.sample(u, v);
    }

    __device__ vec3 sample_normal(float u, float v) {
        if (!normal_map.is_set())
            return vec3(0.0f, 0.0f, 1.0f);
        vec3 ret = 2.0f * normal_map.sample(u, v) - 1.0f;
        return ret.normalized();
    }

    // base properties
    vec3 albedo = vec3(0.8f);
    vec3 specular = vec3(0.0f);
    float emission = 0.0f;
    float smoothness = 0.0f;
    bool smooth_shading = false;
    bool on_device = false;

    // textures
    ColorTexture albedo_map;
    ColorTexture specular_map;
    ColorTexture normal_map;
    GrayscaleTexture smoothness_map;
};

#endif