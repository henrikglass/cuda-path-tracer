#ifndef GEOMETRY_H
#define GEOMETRY_H

#define EPSILON 0.00001f

#include <string>
#include <float.h>
#include "material.cuh"
#include "vector.cuh"

enum Shape {SPHERE, TRIANGLE_MESH};

struct Ray {
    __device__ __host__ Ray(vec3 origin, vec3 direction) {
        this->origin    = origin;
        this->direction = direction;
    }
    vec3 origin;
    vec3 direction;
};

struct Vertex {
    __host__ Vertex() {}
    __host__ Vertex(vec3 position, vec3 normal) {
        this->position  = position;
        this->normal    = normal;
    }
    vec3 position, normal;
};

struct Triangle {
    __host__ Triangle() {}
    __host__ Triangle(int idx_a, int idx_b, int idx_c) {
        this->idx_a = idx_a;
        this->idx_b = idx_b;
        this->idx_c = idx_c;
    }
    int idx_a, idx_b, idx_c;
};

struct AABB {
    AABB() {}
    AABB(vec3 min, vec3 max) {
        this->min = min;
        this->max = max;
    }
    __host__ void recalculate(Vertex *vertices, int n_vertices);
    __device__ bool intersects(const Ray &ray);
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
    Triangle *triangles   = nullptr;
    Vertex *vertices      = nullptr;
    Triangle *d_triangles = nullptr;
    Vertex *d_vertices    = nullptr;
    long triangles_size;
    long vertices_size;
    AABB aabb;

    // for sphere case:
    float radius;

    // for general case:
    vec3 center;

    // misc. functions
    __device__ bool get_closest_sphere_intersection(const Ray &ray, Intersection &bestHit);
    __device__ bool get_closest_triangle_mesh_intersection(const Ray &ray, Intersection &bestHit);
public:
    Shape shape; // Workaround for losing virtual inheritance with cuda. 

    /*
     * create triangle mesh entity from path to .obj. Providing your own material.
     */
    Entity(const std::string &path, const Material &material);
    /*
     * create sphere entity from coordinate and radius. Providing your own material.
     */
    Entity(const vec3 &center, float radius, const Material &material);
    
    // memory management
    void copy_to_device();
    void free_from_device();

    // misc. functions
    __device__ bool get_closest_intersection(const Ray &ray, Intersection &bestHit);
    __host__ void scale(float factor);
    __host__ void translate(vec3 delta);
    __host__ void print();

    Material material;

    // TODO move to private
};

__device__ bool intersects_aabb(
        float min_x,
        float min_y,
        float min_z,
        float max_x,
        float max_y,
        float max_z,
        const Ray &ray
);

__device__ bool get_closest_intersection_in_scene(
        const Ray &ray, 
        Entity *entities, 
        int n_entities, 
        Intersection &bestHit
);

#endif