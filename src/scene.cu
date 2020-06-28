#include "scene.cuh"
#include "util.cuh"

Scene::Scene() {
    this->d_entities = nullptr;
}

Scene::~Scene() {
    this->free_from_device();
}

void Scene::set_hdri(const std::string &path) {
    HDRLoader loader;
    if(!loader.load(path.c_str(), this->hdri)) {
        std::cerr << "Error parsing .hdr file, or file doesn't exist." << std::endl;
        exit(1);
    }
    has_hdri = true;
}

void Scene::use_hdri_smoothing(bool b) {
    this->use_smooth_hdri = b;
}

void Scene::add_entity(Entity *entity) {
    this->entities.push_back(entity);
}

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