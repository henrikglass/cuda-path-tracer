#ifndef TEXTURE_H
#define TEXTURE_H

#include "vector.cuh"
#include "util.cuh"

struct Texture {
    //@Incomplete Maybe we want to flip textures along y.
    Texture() {}
    ~Texture();
    void free_from_device();
    __host__ __device__ bool is_set() { return this->data != nullptr; }
    unsigned char *data = nullptr;
    unsigned char *d_data = nullptr;
    ivec2 resolution;
};

struct ColorTexture : Texture {
    ColorTexture()  {}
    void copy_to_device();
    void set(const std::string &path);
    __device__ vec3 sample(float u, float v) {
        int x = u * resolution.x;
        int y = (1.0f - v) * resolution.y;
        int idx = 3 * (y * resolution.x + x);
        return vec3(
            float(this->d_data[idx]) / 255.0f,
            float(this->d_data[idx + 1]) / 255.0f,
            float(this->d_data[idx + 2]) / 255.0f
        );
    }
};

struct GrayscaleTexture : Texture {
    GrayscaleTexture() {}
    void copy_to_device();
    void set(const std::string &path);
    __device__ float sample(float u, float v) {
        int x = u * resolution.x;
        int y = (1.0f - v) * resolution.y;
        int idx = y * resolution.x + x;
        return float(this->d_data[idx]) / 255.0f;
    }
};

#endif