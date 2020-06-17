#include "geometry.cuh"

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

__host__ 
bool AABB::contains_triangle(vec3 v0, vec3 v1, vec3 v2) {
    return triangle_inside_aabb(
            this->min.x, this->min.y, this->min.z, 
            this->max.x, this->max.y, this->max.z,
            v0, v1, v2
    );
}

__host__ 
inline bool triangle_inside_aabb(
        float min_x,
        float min_y,
        float min_z,
        float max_x,
        float max_y,
        float max_z,
        vec3 v0,
        vec3 v1,
        vec3 v2
) {
    return  inside_aabb(min_x, min_y, min_z, max_x, max_y, max_z, v0) &&
            inside_aabb(min_x, min_y, min_z, max_x, max_y, max_z, v1) &&
            inside_aabb(min_x, min_y, min_z, max_x, max_y, max_z, v2);
}

__host__ 
inline bool inside_aabb(
        float min_x,
        float min_y,
        float min_z,
        float max_x,
        float max_y,
        float max_z,
        const vec3 &point
) {
    return  point.x > min_x && point.x < max_x &&
            point.y > min_y && point.y < max_y &&
            point.z > min_z && point.z < max_z;
}