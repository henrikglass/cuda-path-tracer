#include "scene.cuh"
#include "util.cuh"

Scene::Scene() {
    this->d_entities = nullptr;
}

void Scene::addEntity(Entity entity) {
    this->entities.push_back(entity);
}

void Scene::copyToDevice() {
    for (Entity e : this->entities) {
        e.moveToDevice();
    }
    long size = this->entities.size()*sizeof(Entity);
    gpuErrchk(cudaMalloc(&this->d_entities, size));
    cudaMemcpy(this->d_entities, &(this->entities[0]), size, cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError());
}

void Scene::freeFromDevice() {

}