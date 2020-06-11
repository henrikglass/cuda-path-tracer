#ifndef GEOMETRY_H
#define GEOMETRY_H

#define EPSILON 0.00001f

#include <string>
#include <float.h>
#include "material.cuh"

enum Shape {SPHERE, TRIANGLE_MESH};

struct Ray {
    __device__ __host__ Ray(vec3 origin, vec3 direction) {
        this->origin = origin;
        this->direction = direction;
    }
    vec3 origin;
    vec3 direction;
};

struct Triangle {
    vec3 a, b, c;
    __device__ __host__ Triangle(vec3 a, vec3 b, vec3 c) {
        this->a = a;
        this->b = b;
        this->c = c;
    }
};

struct AABB {
    AABB() {}
    AABB(vec3 min, vec3 max) {
        this->min = min;
        this->max = max;
    }
    vec3 min;
    vec3 max;
};

class Entity;

struct Intersection {
    __device__ Intersection() {
        this->distance = FLT_MAX;
    }
    vec3 position;
    vec3 normal;
    float distance;
    Entity *entity;
};

class Entity {
private:

    // for triangle mesh case:
    Triangle *triangles = nullptr;
    vec3 *vertices = nullptr;
    AABB aabb;

    //// for sphere case:
    //vec3 center;
    //float radius;

    // misc. functions
    __device__ bool get_closest_sphere_intersection(const Ray &ray, Intersection &bestHit);
    __device__ bool get_closest_triangle_mesh_intersection(const Ray &ray, Intersection &bestHit);
public:
    Shape shape; // Workaround for losing virtual inheritance with cuda. 

    // constructors
    Entity(const std::string &path, const Material &material);            // create triangle mesh entity from path to .obj.
    Entity(const vec3 &center, float radius, const Material &material);   // create sphere entity from coordinate and radius.
    
    // cuda specifics
    void move_to_device();
    void free_from_device();

    // misc. functions
    __device__ bool get_closest_intersection(const Ray &ray, Intersection &bestHit);

    //// for sphere case:
    vec3 center;
    float radius;

    Material material;
};

__device__ bool get_closest_intersection_in_scene(
        const Ray &ray, 
        Entity *entities, 
        int n_entities, 
        Intersection &bestHit
);

#endif