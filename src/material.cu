#include "material.cuh"
#include <assert.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Material::Material(vec3 albedo, vec3 specular, float emission, float smoothness, bool smooth_shading) {
    this->albedo             = albedo;
    this->specular           = specular;
    this->emission           = emission;
    this->smoothness         = smoothness;
    this->smooth_shading     = smooth_shading;
}

Material::~Material() {
    this->free_from_device();
    if (this->albedo_map != nullptr) {
        stbi_image_free(this->albedo_map);
    }
}

void Material::set_albedo_map(const std::string &path, bool y_flipped) {
    int n;
    this->albedo_map = stbi_load(path.c_str(), &(this->texture_res.x), &(this->texture_res.y), &n, 3);
    assert(n == 3);
    this->has_albedo_map = true;
    this->y_flip = y_flipped;
}

__host__ 
void Material::copy_to_device() {
    if (this->on_device)
        return;
    
    // copy albedo map
    if (this->albedo_map != nullptr) {
        long size = sizeof(unsigned char) * texture_res.x * texture_res.y * 3; // assume 3 channels per pixel
        gpuErrchk(cudaMalloc(&this->d_albedo_map, size));
        cudaMemcpy(this->d_albedo_map, this->albedo_map, size, cudaMemcpyHostToDevice);
        gpuErrchk(cudaPeekAtLastError());
    }

    this->on_device = true;
}

__host__ 
void Material::free_from_device() {
    if (!this->on_device)
        return;
    
    // free albedo map
    if (this->d_albedo_map != nullptr) {
        gpuErrchk(cudaFree(this->d_albedo_map));
        this->d_albedo_map = nullptr;
    }

    this->on_device = false;
}