#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <string>
#include "material.cuh"

enum Shape {SPHERE, TRIANGLE_MESH};

struct Ray {
    Ray(vec3 origin, vec3 direction) {
        this->origin = origin;
        this->direction = direction;
    }
    vec3 origin;
    vec3 direction;
};

struct Triangle {
    vec3 a, b, c;
    Triangle(vec3 a, vec3 b, vec3 c) {
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

struct Intersection {
    vec3 position;
    vec3 normal;
    float distance;
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
    bool getClosestSphereIntersection(const Ray &ray, Intersection &bestHit);
    bool getClosestTriangleMeshIntersection(const Ray &ray, Intersection &bestHit);
public:
    Shape shape; // Workaround for losing virtual inheritance with cuda. 

    // constructors
    Entity(const std::string &path, const Material &material);            // create triangle mesh entity from path to .obj.
    Entity(const vec3 &center, float radius, const Material &material);   // create sphere entity from coordinate and radius.
    
    // cuda specifics
    void moveToDevice();
    void freeFromDevice();

    // misc. functions
    bool getClosestIntersection(const Ray &ray, Intersection &bestHit);

    //// for sphere case:
    vec3 center;
    float radius;

    Material material;
};

#endif