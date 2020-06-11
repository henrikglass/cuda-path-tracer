#include "geometry.cuh"
#include <math.h>

Entity::Entity(const std::string &path, const Material &material) {
    // TODO implement
}

Entity::Entity(const vec3 &center, float radius, const Material &material) {
    this->shape     = SPHERE;
    this->center    = center;
    this->radius    = radius;
    this->material  = material;
}

void Entity::move_to_device() {
    if (this->shape == SPHERE)
        return;

    // TODO implement mesh case
}

void Entity::free_from_device() {
    if (this->shape == SPHERE)
        return;

    // TODO implement mesh case
}

bool Entity::get_closest_intersection(const Ray &ray, Intersection &bestHit) {
    switch(this->shape) {
        case SPHERE:
            return get_closest_sphere_intersection(ray, bestHit);
        case TRIANGLE_MESH:
            return get_closest_triangle_mesh_intersection(ray, bestHit);
        default:
            return false;
    }
}

bool Entity::get_closest_sphere_intersection(const Ray &ray, Intersection &bestHit) {
    vec3 d = ray.origin - this->center;
    float p1 = -dot(ray.direction, d);
    float p2sqr = p1 * p1 - dot(d,d) + this->radius * this->radius;
    if (p2sqr < 0)
        return false;
    float p2 = sqrt(p2sqr);
    float t = p1 - p2 > 0 ? p1 - p2 : p1 + p2;
    if (t > 0 && t < bestHit.distance)
    {
        bestHit.distance = t;
        bestHit.position = ray.origin + t * ray.direction;
        bestHit.normal = bestHit.position - this->center;
        bestHit.normal.normalize();
        return true;
    }
    return false;
}

bool Entity::get_closest_triangle_mesh_intersection(const Ray &ray, Intersection &bestHit) {
    // TODO Implement
    return false;
}