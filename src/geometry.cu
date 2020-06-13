#include "geometry.cuh"
#include "OBJ_Loader.h"
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include "util.cuh"

//#pragma hd_warning_disable
//#include "glm/glm/glm.hpp"

/************************************************************************************/
/*                                    Constructors                                  */
/************************************************************************************/

Entity::Entity(const std::string &path, const Material &material) {
    objl::Loader ojb_loader;
    bool err_nil = ojb_loader.LoadFile(path);

    // Check for errors or unwelcome extra meshes
    if (!err_nil) {
        std::cerr << "Error loading " << path << std::endl;
        exit(EXIT_FAILURE);
    }
    if (ojb_loader.LoadedMeshes.size() != 1) {
        std::cerr << path << " Does not consist of only 1 mesh" << std::endl;
        exit(EXIT_FAILURE);
    }

    // otherwise get the mesh
    objl::Mesh mesh = ojb_loader.LoadedMeshes[0];

    // allocate space
    this->n_vertices  = mesh.Vertices.size();
    this->n_triangles = mesh.Indices.size() / 3;
    this->vertices  = new Vertex[this->n_vertices];
    this->triangles = new Triangle[this->n_triangles];
    
    // For AABB
    vec3 min(FLT_MAX, FLT_MAX, FLT_MAX);
    vec3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    // get vertices
    for (int i = 0; i < mesh.Vertices.size(); i++) {
        vec3 position(
                mesh.Vertices[i].Position.X,
                mesh.Vertices[i].Position.Y,
                mesh.Vertices[i].Position.Z
        );
        vec3 normal(
                mesh.Vertices[i].Normal.X,
                mesh.Vertices[i].Normal.Y,
                mesh.Vertices[i].Normal.Z
        );
        normal.normalize();

        // update min, max
        min.x = (position.x < min.x) ? position.x : min.x;
        min.y = (position.y < min.y) ? position.y : min.y;
        min.z = (position.z < min.z) ? position.z : min.z;
        max.x = (position.x > max.x) ? position.x : max.x;
        max.y = (position.y > max.y) ? position.y : max.y;
        max.z = (position.z > max.z) ? position.z : max.z;

        this->vertices[i] = Vertex(position, normal);
    }

    // get triangles
    for (int i = 0; i < mesh.Indices.size(); i += 3) {
        this->triangles[i / 3] = Triangle(
                mesh.Indices[i],
                mesh.Indices[i + 1],
                mesh.Indices[i + 2]
        );
    }

    std::cout << "KALAS" << std::endl;
    std::cout << this->vertices[2727].normal << std::endl;
    std::cout << this->vertices[2728].normal << std::endl;
    std::cout << this->vertices[2729].normal << std::endl;

    // debug print all triangle vertices
    /*for(int i = 0; i < this->n_triangles; i++) {
        std::cout << "tri: " << i << std::endl;
        std::cout << this->vertices[this->triangles[i].idx_a].position << std::endl;
        std::cout << this->vertices[this->triangles[i].idx_b].position << std::endl;
        std::cout << this->vertices[this->triangles[i].idx_c].position << std::endl;
    }*/

    // create AABB
    this->aabb = AABB(min, max);

    // set center
    this->center = vec3(
            (min.x + max.x) / 2,
            (min.y + max.y) / 2,
            (min.z + max.z) / 2
    );

    // set shape
    this->shape = TRIANGLE_MESH;
}

Entity::Entity(const vec3 &center, float radius, const Material &material) {
    this->shape     = SPHERE;
    this->center    = center;
    this->radius    = radius;
    this->material  = material;
}

/************************************************************************************/
/*                                 Misc. fucntions                                  */
/************************************************************************************/
void AABB::recalculate(Vertex *vertices, int n_vertices) {
    vec3 min(FLT_MAX, FLT_MAX, FLT_MAX);
    vec3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (int i = 0; i < n_vertices; i++) {
        vec3 position = vertices[i].position;
        min.x = (position.x < min.x) ? position.x : min.x;
        min.y = (position.y < min.y) ? position.y : min.y;
        min.z = (position.z < min.z) ? position.z : min.z;
        max.x = (position.x > max.x) ? position.x : max.x;
        max.y = (position.y > max.y) ? position.y : max.y;
        max.z = (position.z > max.z) ? position.z : max.z;
    }
    
    this->min = min;
    this->max = max;
}

void Entity::print() {
    std::cout << "center: " << this->center << std::endl;
    std::cout << "x: " << this->aabb.min.x << " to " << this->aabb.max.x << std::endl;
    std::cout << "y: " << this->aabb.min.y << " to " << this->aabb.max.y << std::endl;
    std::cout << "z: " << this->aabb.min.z << " to " << this->aabb.max.z << std::endl;
}

/************************************************************************************/
/*                            Geometric transformations                             */
/************************************************************************************/

void Entity::scale(float factor) {
    if (this->shape == SPHERE) {
        radius *= factor;
        return;
    }

    if (this->shape == TRIANGLE_MESH) {
        int v_size = this->n_vertices;
        for(int i = 0; i < v_size; i++) {
            vec3 pos = vertices[i].position;
            pos = ((pos - this->center) * factor) + this->center;
            vertices[i].position = pos;
        }

        // recalculate aabb
        this->aabb.recalculate(this->vertices, this->n_vertices);
    }
}

void Entity::translate(vec3 delta) {
    // move center for all shapes
    this->center = this->center + delta;

    if (this->shape == TRIANGLE_MESH) {
        int v_size = this->n_vertices;
        for(int i = 0; i < v_size; i++) {
            vec3 pos = vertices[i].position;
            pos = pos + delta;
            vertices[i].position = pos;
        }

        // recalculate aabb
        this->aabb.recalculate(this->vertices, this->n_vertices);
    }
}

void Entity::rotate(vec3 rot) {
    if (this->shape == SPHERE)
        return;
    
    // rotate on x
    for (int i = 0; i < this->n_vertices; i++) {
        vec3 v = this->vertices[i].position - this->center;
        v = vec3(
            v.x,
            v.y*cos(rot.x) - v.z*sin(rot.x),
            v.y*sin(rot.x) - v.z*cos(rot.x)
        );
        this->vertices[i].position = v + this->center;
    }

    // recalculate aabb
    this->aabb.recalculate(this->vertices, this->n_vertices);

    // TODO rotate on y and z. Preferable not one at a time.
}

/************************************************************************************/
/*                                Memory management                                 */
/************************************************************************************/

void Entity::copy_to_device() {
    if (this->shape == SPHERE)
        return;

    if (this->shape == TRIANGLE_MESH) {
        // copy vertices
        long vertices_size = this->n_vertices * sizeof(Vertex);
        gpuErrchk(cudaMalloc(&this->d_vertices, vertices_size));
        cudaMemcpy(this->d_vertices, this->vertices, vertices_size, cudaMemcpyHostToDevice);
        gpuErrchk(cudaPeekAtLastError());

        // copy triangles
        long triangles_size = this->n_triangles * sizeof(Triangle);
        gpuErrchk(cudaMalloc(&this->d_triangles, triangles_size));
        cudaMemcpy(this->d_triangles, this->triangles, triangles_size, cudaMemcpyHostToDevice);
        gpuErrchk(cudaPeekAtLastError());
    }
}

void Entity::free_from_device() {
    if (this->shape == SPHERE)
        return;

    // TODO implement mesh case
}


/************************************************************************************/
/*                            Intersection functions                                */
/************************************************************************************/

__device__
bool get_closest_intersection_in_scene(const Ray &ray, Entity *entities, int n_entities, Intersection &is) {
    bool is_hit = false;
    for (int i = 0; i < n_entities; i++) {
        is_hit = entities[i].get_closest_intersection(ray, is) || is_hit;
    }

    // if hit entity has smooth_shading enabled, adjust the normal
    /*Triangle *tr = is.triangle;
    Entity *e = is.entity;
    if (is_hit && tr != nullptr && e->material.smooth_shading) {
        float u = is.u;
        float v = is.v;
        float w = 1.0f - (u + v);
        printf("idxs: %d %d %d\n", tr->idx_a, tr->idx_b, tr->idx_c);
        vec3 v0_normal = e->d_vertices[tr->idx_a].normal;
        vec3 v1_normal = e->d_vertices[tr->idx_b].normal;
        vec3 v2_normal = e->d_vertices[tr->idx_c].normal;
        printf("v0 normal: (%g, %g, %g)\n", v0_normal.x, v0_normal.y, v0_normal.z);
        printf("v1 normal: (%g, %g, %g)\n", v1_normal.x, v1_normal.y, v1_normal.z);
        printf("v2 normal: (%g, %g, %g)\n", v2_normal.x, v2_normal.y, v2_normal.z);
        printf("flat normal: (%g, %g, %g)\n", is.normal.x, is.normal.y, is.normal.z);
        //is.normal = -(u * v1_normal + v * v2_normal + w * v0_normal); // pure guess
        //is.normal = -(u * v2_normal + v * v1_normal + w * v0_normal); // pure guess
        //is.normal = u * v0_normal + v * v2_normal + w * v1_normal; // pure guess
        //is.normal = u * v2_normal + v * v0_normal + w * v1_normal; // pure guess
        //is.normal = u * v1_normal + v * v0_normal + w * v2_normal; // pure guess
        //is.normal = u * v0_normal + v * v1_normal + w * v2_normal; // pure guess
        //is.normal.normalize();
    }*/

    return is_hit;
}

__device__
bool Entity::get_closest_intersection(const Ray &ray, Intersection &bestHit) {
    switch (this->shape) {
        case SPHERE:
            return get_closest_sphere_intersection(ray, bestHit);
        case TRIANGLE_MESH:
            return get_closest_triangle_mesh_intersection(ray, bestHit);
        default:
            return false;
    }
}

__device__
bool Entity::get_closest_sphere_intersection(const Ray &ray, Intersection &bestHit) {
    vec3 d = ray.origin - this->center;
    float p1 = -dot(ray.direction, d);
    float p2sqr = p1 * p1 - dot(d,d) + this->radius * this->radius;
    if (p2sqr < 0)
        return false;
    float p2 = sqrtf(p2sqr); // sqrt(p2sqr)
    float t = p1 - p2 > 0 ? p1 - p2 : p1 + p2;
    if (t > 0 && t < bestHit.distance)
    {
        bestHit.distance = t;
        bestHit.position = ray.origin + t * ray.direction;
        bestHit.normal = bestHit.position - this->center;
        bestHit.normal.normalize();
        bestHit.entity = this;
        return true;
    }
    return false;
}

__device__
bool Entity::get_closest_triangle_mesh_intersection(const Ray &ray, Intersection &bestHit) {
    if (!this->aabb.intersects(ray))
        return false;
    bool hit = false;
    for (int i = 0; i < this->n_triangles; i++) {
        hit = intersects_triangle(&(this->d_triangles[i]), bestHit, ray) || hit;
    }
    return hit;
}


/*
 * Tomas Akenine-Möller and Ben Trumbore's algorithm.
 *
 * http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/pubs/raytri_tam.pdf
 */
__device__
bool Entity::intersects_triangle(Triangle *triangle, Intersection &bestHit, const Ray &ray) {
    vec3 v0 = this->d_vertices[triangle->idx_a].position;
    vec3 v1 = this->d_vertices[triangle->idx_b].position;
    vec3 v2 = this->d_vertices[triangle->idx_c].position;
    vec3 e1, e2, pvec, tvec, qvec;
    float t, u, v, det, inv_det;

    e1 = v1 - v0;
    e2 = v2 - v0;

    pvec = cross(ray.direction, e2);
    det = dot(e1, pvec);
    if (fabs(det) < EPSILON) 
        return false;
    
    inv_det = 1.0f / det;
    tvec = ray.origin - v0;
    u = dot(tvec, pvec) * inv_det;
    if (u < 0.0f || u > 1.0f)
        return false;

    qvec = cross(tvec, e1);
    v = dot(ray.direction, qvec) * inv_det;
    if (v < 0.0f || u + v > 1.0f)
        return false;

    t = dot(e2, qvec) * inv_det; 

    if(t > 0 && t < bestHit.distance) {
        bestHit.distance = t;
        bestHit.position = ray.origin + t * ray.direction;
        bestHit.normal = cross(e1, e2).normalized(); // TODO SMOOTH SHADING
        bestHit.entity = this;
        bestHit.triangle = triangle;
        bestHit.u = u;
        bestHit.v = v;
        return true;
    }

    return false;
}

__device__ 
bool AABB::intersects(const Ray &ray) {
    return intersects_aabb(
            this->min.x,
            this->min.y,
            this->min.z,
            this->max.x,
            this->max.y,
            this->max.z,
            ray
    );
}

__device__
bool intersects_aabb(
        float min_x,
        float min_y,
        float min_z,
        float max_x,
        float max_y,
        float max_z,
        const Ray &ray
) {

    float tx1 = (min_x - ray.origin.x)*(1.0f / ray.direction.x);
    float tx2 = (max_x - ray.origin.x)*(1.0f / ray.direction.x);
    float ty1 = (min_y - ray.origin.y)*(1.0f / ray.direction.y);
    float ty2 = (max_y - ray.origin.y)*(1.0f / ray.direction.y);
    float tz1 = (min_z - ray.origin.z)*(1.0f / ray.direction.z);
    float tz2 = (max_z - ray.origin.z)*(1.0f / ray.direction.z);

    float tmin = fminf(tx1, tx2);
    float tmax = fmaxf(tx1, tx2);
    tmin = fmaxf(tmin, fminf(ty1, ty2));
    tmax = fminf(tmax, fmaxf(ty1, ty2));
    tmin = fmaxf(tmin, fminf(tz1, tz2));
    tmax = fminf(tmax, fmaxf(tz1, tz2));
 
    return tmin < tmax;
}