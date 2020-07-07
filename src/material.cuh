#ifndef MATERIAL_H
#define MATERIAL_H

#include "vector.cuh"
#include "util.cuh"

struct Material {
    Material() {}
    ~Material();
    Material(vec3 albedo, vec3 specular, float emission, float smoothness, bool smooth_shading);

    void set_albedo_map(const std::string &path, bool y_flipped = false);

    // memory management
    __host__ void copy_to_device();
    __host__ void free_from_device();

    // device functions
    __device__ vec3 sample_albedo(float u, float v) {
        if(!this->has_albedo_map)
            return this->albedo;
        
        v = (y_flip) ? v : (1.0f - v);
        int x = u * this->texture_res.x;
        int y = v * this->texture_res.y;
        int pixel_idx = 3 * (y * this->texture_res.x + x);
        return vec3(
                float(this->d_albedo_map[pixel_idx]) / 255.0f,
                float(this->d_albedo_map[pixel_idx + 1]) / 255.0f,
                float(this->d_albedo_map[pixel_idx + 2]) / 255.0f
        );
    }

    // properties
    vec3 albedo = vec3(0.8f);
    vec3 specular = vec3(0.0f);
    float emission = 0.0f;
    float smoothness = 0.0f;
    bool smooth_shading = false;
    bool on_device = false;

    // texture stuff
    ivec2 texture_res = ivec2(0,0);
    bool y_flip                   = false;
    bool has_albedo_map           = false;
    unsigned char *albedo_map     = nullptr;
    unsigned char *d_albedo_map   = nullptr;
};

#endif