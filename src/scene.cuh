#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include "geometry.cuh"

class Scene {
public:
    Scene();
    std::vector<Entity> entities;
    Entity *d_entities;
    void addEntity(Entity entity);
    void copyToDevice();
    void freeFromDevice();

};

#endif