#ifndef GEOMETRY_H
#define GEOMETRY_H

#define MAX_OCTREE_DEPTH 6
#define EPSILON 0.000000001f
#define MAX_RDIR 1000000.0f
#define AABB_PADDING 0.0001f

#include <string>
#include <float.h>
#include "material.cuh"
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
        // @Incomplete this only works with positive sign
        fracs.x = fmaxf(fminf(1.0f / this->direction.x, MAX_RDIR), -MAX_RDIR);
        fracs.y = fmaxf(fminf(1.0f / this->direction.y, MAX_RDIR), -MAX_RDIR);
        fracs.z = fmaxf(fminf(1.0f / this->direction.z, MAX_RDIR), -MAX_RDIR);
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
    __host__ Triangle(
            int idx_a, int idx_b, int idx_c,
            int vt_idx_a, int vt_idx_b, int vt_idx_c
    ) {
        this->idx_a = idx_a;
        this->idx_b = idx_b;
        this->idx_c = idx_c;
        this->vt_idx_a = vt_idx_a;
        this->vt_idx_b = vt_idx_b;
        this->vt_idx_c = vt_idx_c;
    }
    int idx_a, idx_b, idx_c;
    int vt_idx_a, vt_idx_b, vt_idx_c;
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
    __host__ bool intersects_triangle(vec3 u0, vec3 u1, vec3 u2);
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
private:

    // for triangle mesh case:
    Octree *octree      = nullptr;
    Octree *d_octree    = nullptr;
    Vertex *vertices    = nullptr;
    Triangle *triangles = nullptr;
    vec2 *uvs           = nullptr;
    size_t n_triangles;
    size_t n_vertices;
    size_t n_uvs;
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
    Entity(const std::string &path, Material *material);
    /*
     * create sphere entity from coordinate and radius. Providing your own material.
     */
    Entity(vec3 center, float radius, Material *material);
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
    __device__ friend bool trace(
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

    Material *material;
    Material *d_material;

    bool on_device = false;
    Vertex *d_vertices      = nullptr;
    Triangle *d_triangles   = nullptr;
    vec2 *d_uvs             = nullptr;
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
        vec3 point
);



#endif
