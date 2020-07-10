#include "material.cuh"
#include <assert.h>

/*
 * Material
 */
Material::Material(vec3 albedo, vec3 specular, float emission, float smoothness, bool smooth_shading) {
    this->albedo             = albedo;
    this->specular           = specular;
    this->emission           = emission;
    this->smoothness         = smoothness;
    this->smooth_shading     = smooth_shading;
}

Material::~Material() {
    this->free_from_device();
}

__host__ 
void Material::copy_to_device() {
    if (this->on_device)
        return;

    // copy textures
    this->albedo_map.copy_to_device();
    this->specular_map.copy_to_device();
    this->smoothness_map.copy_to_device();
    this->normal_map.copy_to_device();

    this->on_device = true;
}

__host__ 
void Material::free_from_device() {
    if (!this->on_device)
        return;
    
    // free textures
    this->albedo_map.free_from_device();
    this->specular_map.free_from_device();
    this->smoothness_map.free_from_device();
    this->normal_map.free_from_device();

    this->on_device = false;
}