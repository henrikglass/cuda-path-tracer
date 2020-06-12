#include "scene.cuh"
#include "util.cuh"

Scene::Scene() {
    this->d_entities = nullptr;
}

void Scene::add_entity(Entity entity) {
    this->entities.push_back(entity);
}

void Scene::copy_to_device() {
    for (Entity e : this->entities) {
        e.copy_to_device();
    }
    long size = this->entities.size()*sizeof(Entity);
    gpuErrchk(cudaMalloc(&this->d_entities, size));
    cudaMemcpy(this->d_entities, &(this->entities[0]), size, cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError());
}

void Scene::free_from_device() {

}