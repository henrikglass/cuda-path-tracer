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

void Material::flip_texture_y() {
    this->y_flip = !this->y_flip;
}

void Material::set_albedo_map(const std::string &path) {
    int n;
    this->albedo_map = stbi_load(path.c_str(), &(this->albedo_map_res.x), &(this->albedo_map_res.y), &n, 3);
    this->textures_set |= ALBEDO_MAP_SET;
}

void Material::set_specular_map(const std::string &path) {
    int n;
    this->specular_map = stbi_load(path.c_str(), &(this->specular_map_res.x), &(this->specular_map_res.y), &n, 3);
    this->textures_set |= SPECULAR_MAP_SET;
}

void Material::set_smoothness_map(const std::string &path) {
    int n;
    this->smoothness_map = stbi_load(path.c_str(), &(this->smoothness_map_res.x), &(this->smoothness_map_res.y), &n, 1);
    this->textures_set |= SMOOTHNESS_MAP_SET;
}

void Material::set_normal_map(const std::string &path) {
    int n;
    this->normal_map = stbi_load(path.c_str(), &(this->normal_map_res.x), &(this->normal_map_res.y), &n, 3);
    this->textures_set |= NORMAL_MAP_SET;
}

__host__ 
void Material::copy_to_device() {
    if (this->on_device)
        return;
    
    // copy textures
    if (this->textures_set & ALBEDO_MAP_SET) {
        long size = sizeof(unsigned char) * albedo_map_res.x * albedo_map_res.y * 3; // assume 3 channels per pixel
        gpuErrchk(cudaMalloc(&this->d_albedo_map, size));
        cudaMemcpy(this->d_albedo_map, this->albedo_map, size, cudaMemcpyHostToDevice);
        gpuErrchk(cudaPeekAtLastError());
    }
    if (this->textures_set & SPECULAR_MAP_SET) {
        long size = sizeof(unsigned char) * specular_map_res.x * specular_map_res.y * 3; // assume 3 channels per pixel
        gpuErrchk(cudaMalloc(&this->d_specular_map, size));
        cudaMemcpy(this->d_specular_map, this->specular_map, size, cudaMemcpyHostToDevice);
        gpuErrchk(cudaPeekAtLastError());
    }
    if (this->textures_set & SMOOTHNESS_MAP_SET) {
        long size = sizeof(unsigned char) * smoothness_map_res.x * smoothness_map_res.y; // assume 1 channels per pixel
        gpuErrchk(cudaMalloc(&this->d_smoothness_map, size));
        cudaMemcpy(this->d_smoothness_map, this->smoothness_map, size, cudaMemcpyHostToDevice);
        gpuErrchk(cudaPeekAtLastError());
    }
    if (this->textures_set & NORMAL_MAP_SET) {
        long size = sizeof(unsigned char) * normal_map_res.x * normal_map_res.y * 3; // assume 3 channels per pixel
        gpuErrchk(cudaMalloc(&this->d_normal_map, size));
        cudaMemcpy(this->d_normal_map, this->normal_map, size, cudaMemcpyHostToDevice);
        gpuErrchk(cudaPeekAtLastError());
    }

    this->on_device = true;
}

__host__ 
void Material::free_from_device() {
    if (!this->on_device)
        return;
    
    // free textures
    if (this->d_albedo_map != nullptr) {
        gpuErrchk(cudaFree(this->d_albedo_map));
        this->d_albedo_map = nullptr;
    }
    if (this->d_specular_map != nullptr) {
        gpuErrchk(cudaFree(this->d_specular_map));
        this->d_specular_map = nullptr;
    }
    if (this->d_smoothness_map != nullptr) {
        gpuErrchk(cudaFree(this->d_smoothness_map));
        this->d_smoothness_map = nullptr;
    }
    if (this->normal_map != nullptr) {
        gpuErrchk(cudaFree(this->d_normal_map));
        this->normal_map = nullptr;
    }

    this->on_device = false;
}