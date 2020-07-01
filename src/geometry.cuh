#ifndef GEOMETRY_H
#define GEOMETRY_H

#define MAX_OCTREE_DEPTH 7
#define EPSILON 0.00000001f
#define MAX_RDIR 1000000.0f
#define AABB_PADDING 0.0001f

#include <string>
#include <float.h>
#include "material.h"
#include <vector>
#include "vector.cuh"

enum Shape {SPHERE, TRIANGLE_MESH};

struct Ray {
    __host__ __device__ Ray(vec3 origin, vec3 direction) {
        this->origin    = origin;
        this->direction = direction;
        recalc_fracs();
    }
    __host__ __device__ void recalc_fracs() {
        fracs.x = fminf(1.0f / this->direction.x, MAX_RDIR);
        fracs.y = fminf(1.0f / this->direction.y, MAX_RDIR);
        fracs.z = fminf(1.0f / this->direction.z, MAX_RDIR);
    }
    vec3 origin;
    vec3 direction;
    vec3 fracs;
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

class Entity;

struct Intersection {
    __device__ Intersection() {
        this->distance = FLT_MAX;
    }
    vec3 position;
    vec3 normal;
    float distance;
    Entity *entity;
    Triangle *triangle = nullptr;
    float u, v;
};

struct AABB {
    AABB() {}
    AABB(vec3 min, vec3 max) {
        this->min = min - vec3(AABB_PADDING, AABB_PADDING, AABB_PADDING);
        this->max = max + vec3(AABB_PADDING, AABB_PADDING, AABB_PADDING);
    }
    __host__ void recalculate(Vertex *vertices, int n_vertices);
    __host__ bool contains_triangle(vec3 v0, vec3 v1, vec3 v2);
    __host__ bool intersects_triangle(const vec3 &v0, const vec3 &v1, const vec3 &v2);
    __device__ bool intersects(const Ray &ray, const Intersection &bestHit);
    vec3 min;
    vec3 max;
};

struct Octree {
    Octree(AABB aabb, int depth) : children() {
        this->region = aabb;
        this->depth = depth;
        triangle_indices = std::vector<int>();
    }
    ~Octree();
    __host__ void pretty_print(int child_nr);
    __host__ void copy_to_device();
    __host__ void free_from_device();
    __host__ void insert_triangle(vec3 v0, vec3 v1, vec3 v2, size_t triangle_idx);
    __host__ void insert_triangles(Vertex *vertices, Triangle *triangles, size_t n_triangles);
    __device__ bool get_closest_intersection(
            Vertex *vertices, 
            Triangle *triangles, 
            const Ray &ray, 
            Intersection &bestHit,
            Entity *entity
    );
    __device__ bool ray_step(
            const Ray &ray, 
            Intersection &bestHit,
            Entity *entity
    );
    __device__ bool proc_subtree(
            unsigned char a,
            vec3 t0, 
            vec3 t1,
            const Ray &ray, 
            Intersection &bestHit,
            Entity *entity
    );
    Octree *children[8]{nullptr};
    Octree *d_children[8]{nullptr}; // TODO make copy_to_device() place device addresses in same array as host addresses
    std::vector<int> triangle_indices;
    int *d_triangle_indices = nullptr;
    int n_triangle_indices = 0;
    int depth;
    AABB region;
    bool on_device = false;
};

class Entity {
public:

    // for triangle mesh case:
    Octree *d_octree = nullptr;
    Vertex *vertices        = nullptr;
    Vertex *d_vertices      = nullptr;
    Triangle *triangles     = nullptr;
    Triangle *d_triangles   = nullptr;
    size_t n_triangles;
    size_t n_vertices;
    AABB aabb;

    // for sphere case:
    float radius;

    // for general case:
    vec3 center; // not necessarily center of aabb

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
    /*
     * Destruct. 
     */
    ~Entity();
    
    // memory management
    void copy_to_device();
    void free_from_device();

    // Octree
    __host__ void construct_octree();

    // intersection functions
    __device__ bool get_closest_intersection(const Ray &ray, Intersection &bestHit);
    __device__ bool intersects_triangle(Triangle *triangle, Intersection &bestHit, const Ray &ray);
    __device__ friend bool get_closest_intersection_in_scene(
            const Ray &ray, 
            Entity *entities, 
            int n_entities, 
            Intersection &bestHit
    );

    // tranformation functions
    __host__ void scale(float factor);
    __host__ void translate(vec3 delta);
    __host__ void rotate(vec3 rot);

    // misc functions
    __host__ void print();

    Material material;

    bool on_device = false;
    Octree *octree = nullptr;

};

__host__ inline bool triangle_inside_aabb(
        float min_x,
        float min_y,
        float min_z,
        float max_x,
        float max_y,
        float max_z,
        vec3 v0,
        vec3 v1,
        vec3 v2
);

__host__ inline bool inside_aabb(
        float min_x,
        float min_y,
        float min_z,
        float max_x,
        float max_y,
        float max_z,
        const vec3 &point
);



#endif