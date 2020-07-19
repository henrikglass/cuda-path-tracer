#include "texture.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

//
// @Incomplete Doesn't adhere to the rule-of-three.
//

Texture::~Texture() {
    if (this->data != nullptr) {
        stbi_image_free(this->data);
    }
}

/**
 * Frees texture from device memory.
 */
void Texture::free_from_device() {
    if (this->d_data != nullptr) {
        gpuErrchk(cudaFree(this->d_data));
        this->d_data = nullptr;
    }
}

/**
 * Copies color texture to device memory.
 */
void ColorTexture::copy_to_device() {
    if (!this->is_set())
        return;
    long size = sizeof(unsigned char) * resolution.x * resolution.y * 3;
    gpuErrchk(cudaMalloc(&this->d_data, size));
    cudaMemcpy(this->d_data, this->data, size, cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError());
}

/**
 * Copies grayscale texture to device memory.
 */
void GrayscaleTexture::copy_to_device() {
    if (!this->is_set())
        return;
    
    long size = sizeof(unsigned char) * resolution.x * resolution.y;
    gpuErrchk(cudaMalloc(&this->d_data, size));
    cudaMemcpy(this->d_data, this->data, size, cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError());
}

/**
 * Loads an image file and sets it as a color texture.
 * 
 * @param path      a path to a (*.jpg, *.png, *.gif, etc.) image-file.
 */
void ColorTexture::set(const std::string &path) {
    int n;
    this->data = stbi_load(path.c_str(), &(this->resolution.x), &(this->resolution.y), &n, 3);
}

/**
 * Loads an image file and sets it as a color texture.
 * 
 * @param path      a path to a (*.jpg, *.png, *.gif, etc.) image-file.
 */
void GrayscaleTexture::set(const std::string &path) {
    int n;
    this->data = stbi_load(path.c_str(), &(this->resolution.x), &(this->resolution.y), &n, 1);
}
