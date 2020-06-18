#include "scene.cuh"
#include "util.cuh"

Scene::Scene() {
    this->d_entities = nullptr;
}

Scene::~Scene() {
    this->free_from_device();
}

void Scene::add_entity(Entity *entity) {
    this->entities.push_back(entity);
}

void Scene::copy_to_device() {
    for (size_t i = 0; i < this->entities.size(); i++) {
        entities[i]->copy_to_device();
    }

    long size = this->entities.size()*sizeof(Entity);
    gpuErrchk(cudaMalloc(&this->d_entities, size));
    for (size_t i = 0; i < this->entities.size(); i++) {
        cudaMemcpy((this->d_entities + i*sizeof(Entity)), this->entities[i], size, cudaMemcpyHostToDevice);
    }
    gpuErrchk(cudaPeekAtLastError());

    //long size = this->entities.size()*sizeof(Entity);
    //gpuErrchk(cudaMalloc(&this->d_entities, size));
    //cudaMemcpy(this->d_entities, &(this->entities[0]), size, cudaMemcpyHostToDevice);
    //gpuErrchk(cudaPeekAtLastError());
    this->on_device = true;
}

void Scene::free_from_device() {
    if (!this->on_device)
        return;

    for (size_t i = 0; i < this->entities.size(); i++) {
        this->entities[i]->free_from_device();
    }
    gpuErrchk(cudaFree(this->d_entities));

    this->on_device = false;
}