#include <assert.h>

#include "scene.cuh"
#include "util.cuh"

Scene::Scene() {
    this->d_entities = nullptr;
}

Scene::~Scene() {
    this->free_from_device();
}

/**
 * Sets an hdri for the scene.
 * 
 * @param path      a path to an *.hdr image file.
 */
void Scene::set_hdri(const std::string &path) {
    HDRLoader loader;
    if(!loader.load(path.c_str(), this->hdri)) {
        std::cerr << "Error parsing .hdr file, or file doesn't exist." << std::endl;
        exit(1);
    }
    has_hdri = true;
}

/**
 * Sets hdri exposure.
 */
void Scene::set_hdri_exposure(float exposure) {
    this->hdri_exposure = exposure;
}

/**
 * Sets hdri constrast.
 */
void Scene::set_hdri_contrast(float contrast) {
    this->hdri_contrast = contrast;
}

/**
 * Turns hdri smoothing on/off.
 */
void Scene::use_hdri_smoothing(bool b) {
    this->use_smooth_hdri = b;
}

/**
 * Sets hdri rotation.
 */
void Scene::rotate_hdri(float amount) {
    assert(amount >= 0.0f);
    this->hdri_rot_offset = fmod(amount, 1.0f);
}

/**
 * Adds an object to the scene.
 */
void Scene::add_entity(Entity *entity) {
    this->entities.push_back(entity);
}

/**
 * Copies scene to device memory.
 */
void Scene::copy_to_device() {
    if (has_hdri) {
        this->hdri.copy_to_device();
        gpuErrchk(cudaPeekAtLastError());
    }

    for (size_t i = 0; i < this->entities.size(); i++) {
        entities[i]->copy_to_device();
    }

    long size = this->entities.size()*sizeof(Entity);
    gpuErrchk(cudaMalloc(&this->d_entities, size));
    for (size_t i = 0; i < this->entities.size(); i++) {
        cudaMemcpy((this->d_entities + i), this->entities[i], sizeof(Entity), cudaMemcpyHostToDevice);
    }
    gpuErrchk(cudaPeekAtLastError());

    //long size = this->entities.size()*sizeof(Entity);
    //gpuErrchk(cudaMalloc(&this->d_entities, size));
    //cudaMemcpy(this->d_entities, &(this->entities[0]), size, cudaMemcpyHostToDevice);
    //gpuErrchk(cudaPeekAtLastError());
    this->on_device = true;
    this->n_entities = this->entities.size();
}

/**
 * Frees scene from device memory.
 */
void Scene::free_from_device() {
    if (!this->on_device)
        return;
    
    if (has_hdri) {
        this->hdri.free_from_device();
        gpuErrchk(cudaPeekAtLastError());
    }

    for (size_t i = 0; i < this->entities.size(); i++) {
        this->entities[i]->free_from_device();
    }
    gpuErrchk(cudaFree(this->d_entities));

    this->on_device = false;
}
