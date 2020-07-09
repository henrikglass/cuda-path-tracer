#ifndef MATERIAL_H
#define MATERIAL_H

#define ALBEDO_MAP_SET 0b0001
#define SPECULAR_MAP_SET 0b0010
#define SMOOTHNESS_MAP_SET 0b0100
#define NORMAL_MAP_SET 0b1000
#define PIXEL_TO_FLOAT 0.00392156862f

#include "vector.cuh"
#include "util.cuh"

struct Material {
    Material() {}
    ~Material();
    Material(vec3 albedo, vec3 specular, float emission, float smoothness, bool smooth_shading);

    void set_albedo_map(const std::string &path);
    void set_specular_map(const std::string &path);
    void set_smoothness_map(const std::string &path);
    void set_normal_map(const std::string &path);
    void flip_texture_y();
    
    // memory management
    __host__ void copy_to_device();
    __host__ void free_from_device();

    // device functions
    __device__ int get_pixel_idx(float u, float v, const ivec2 &res, int n_channels) {
        v = (y_flip) ? v : (1.0f - v);
        int x = u * res.x;
        int y = v * res.y;
        return n_channels * (y * res.x + x);
    }

    __device__ vec3 sample_albedo(float u, float v) {
        if(!(this->textures_set & ALBEDO_MAP_SET))
            return this->albedo;
        int pixel_idx = get_pixel_idx(u, v, albedo_map_res, 3);
        return vec3(
                float(this->d_albedo_map[pixel_idx]) / 255.0f,
                float(this->d_albedo_map[pixel_idx + 1]) / 255.0f,
                float(this->d_albedo_map[pixel_idx + 2]) / 255.0f
        );
    }

    __device__ vec3 sample_specular(float u, float v) {
        if(!(this->textures_set & SPECULAR_MAP_SET))
            return this->specular;
        int pixel_idx = get_pixel_idx(u, v, specular_map_res, 3);
        return vec3(
                float(this->d_specular_map[pixel_idx]) / 255.0f,
                float(this->d_specular_map[pixel_idx + 1]) / 255.0f,
                float(this->d_specular_map[pixel_idx + 2]) / 255.0f
        );
    }

    __device__ float sample_smoothness(float u, float v) {
        if(!(this->textures_set & SMOOTHNESS_MAP_SET))
            return this->smoothness;
        int pixel_idx = get_pixel_idx(u, v, smoothness_map_res, 1);
        return float(this->d_smoothness_map[pixel_idx]) / 255.0f;
    }

    __device__ vec3 sample_normal(float u, float v) {
        if(!(this->textures_set & NORMAL_MAP_SET))
            return vec3(0.0f, 0.0f, 1.0f);
        int pixel_idx = get_pixel_idx(u, v, normal_map_res, 2);
        return vec3(
                (float(this->d_normal_map[pixel_idx]) * 2.0f - 1.0f) / 255.0f,
                (float(this->d_normal_map[pixel_idx + 1]) * 2.0f - 1.0f) / 255.0f,
                (float(this->d_normal_map[pixel_idx + 2]) * 2.0f - 1.0f) / 255.0f
        ).normalized();
    }

    // base properties
    vec3 albedo = vec3(0.8f);
    vec3 specular = vec3(0.0f);
    float emission = 0.0f;
    float smoothness = 0.0f;
    bool smooth_shading = false;
    bool on_device = false;

    // textures
    ivec2 albedo_map_res            = ivec2(0,0);
    ivec2 specular_map_res          = ivec2(0,0);
    ivec2 smoothness_map_res        = ivec2(0,0);
    ivec2 normal_map_res            = ivec2(0,0);
    bool y_flip                     = false;
    bool flip_normal                = false;
    unsigned char  textures_set     = 0; // bit 0: albedo, bit 1: specular, bit 2: smoothness, bit 3: normal
    unsigned char *albedo_map       = nullptr;
    unsigned char *d_albedo_map     = nullptr;
    unsigned char *specular_map     = nullptr;
    unsigned char *d_specular_map   = nullptr;
    unsigned char *smoothness_map   = nullptr;
    unsigned char *d_smoothness_map = nullptr;
    unsigned char *normal_map       = nullptr;
    unsigned char *d_normal_map     = nullptr;
};

#endif