#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include "geometry.cuh"

class Scene {
public:
    Scene();
    std::vector<Entity> entities;
    Entity *d_entities;
    void add_entity(Entity entity);
    void copy_to_device();
    void free_from_device();

};

#endif