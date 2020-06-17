
#include "geometry.cuh"
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include "util.cuh"

__device__
bool Octree::get_closest_intersection(
        Vertex *vertices, 
        Triangle *triangles, 
        const Ray &ray, 
        Intersection &bestHit,
        Entity *entity
) {
    if(!this->region.intersects(ray))
        return false;
    
    // check intersections for triangle in this node
    bool hit = false;
    for (int i = 0; i < this->n_triangle_indices; i++) {
        int tri_idx = this->d_triangle_indices[i];
        Triangle *triangle = &(triangles[tri_idx]);
        vec3 v0 = vertices[triangle->idx_a].position;
        vec3 v1 = vertices[triangle->idx_b].position;
        vec3 v2 = vertices[triangle->idx_c].position;
        hit = intersect_triangle(v0, v1, v2, triangle, entity, bestHit, ray) || hit;
    }

    // Check children
    for (int i = 0; i < 8; i++) {
        if(this->d_children[i] != nullptr) {
            hit = d_children[i]->get_closest_intersection(vertices, triangles, ray, bestHit, entity) || hit;
        }
    }

    return hit;
}

__device__
bool get_closest_intersection_in_scene(const Ray &ray, Entity *entities, int n_entities, Intersection &is) {
    bool is_hit = false;
    for (int i = 0; i < n_entities; i++) {
        is_hit = entities[i].get_closest_intersection(ray, is) || is_hit;
    }

    // if hit entity has smooth_shading enabled, adjust the normal
    Triangle *tr = is.triangle;
    Entity *e = is.entity;
    if (is_hit && tr != nullptr && e->material.smooth_shading) {
        float u = is.u;
        float v = is.v;
        float w = 1.0f - (u + v);
        vec3 v0_normal = e->d_vertices[tr->idx_a].normal;
        vec3 v1_normal = e->d_vertices[tr->idx_b].normal;
        vec3 v2_normal = e->d_vertices[tr->idx_c].normal;
        is.normal = u * v1_normal + v * v2_normal + w * v0_normal; // pure guess
        is.normal.normalize();
    }

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
    
    // no octree. Check against all triangles.
    if (this->octree == nullptr) {
        //printf("as list");
        bool hit = false;
        for (int i = 0; i < this->n_triangles; i++) {
            hit = intersects_triangle(&(this->d_triangles[i]), bestHit, ray) || hit;
        }
        return hit;
    } else {
        //printf("as octree");
        return this->d_octree->get_closest_intersection(
                this->d_vertices,
                this->d_triangles, 
                ray, 
                bestHit, 
                this
        );
    }
}

__device__
bool Entity::intersects_triangle(Triangle *triangle, Intersection &bestHit, const Ray &ray) {
    vec3 v0 = this->d_vertices[triangle->idx_a].position;
    vec3 v1 = this->d_vertices[triangle->idx_b].position;
    vec3 v2 = this->d_vertices[triangle->idx_c].position;
    return intersect_triangle(v0, v1, v2, triangle, this, bestHit, ray);
}

/*
 * Tomas Akenine-Möller and Ben Trumbore's algorithm.
 *
 * http://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/pubs/raytri_tam.pdf
 */
__device__ 
bool intersect_triangle(
        vec3 v0, 
        vec3 v1, 
        vec3 v2, 
        Triangle *triangle,
        Entity *entity, 
        Intersection &bestHit, 
        const Ray &ray
) {

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
        bestHit.entity = entity;
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