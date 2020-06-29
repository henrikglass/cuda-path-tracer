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
    
    this->min = min - vec3(0.1f, 0.1f, 0.1f); // padding
    this->max = max + vec3(0.1f, 0.1f, 0.1f);
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
bool AABB::intersects_triangle(const vec3 &v0, const vec3 &v1, const vec3 &v2) {
    // check if any vertex is in AABB.
    auto in_aabb = [this](const vec3 &v) {
        return inside_aabb(
            this->min.x, this->min.y, this->min.z, 
            this->max.x, this->max.y, this->max.z, 
            v
        );
    };
    if (in_aabb(v0) || in_aabb(v1) || in_aabb(v2))
        return true;

    // check if any edge intersects AABB.
    Ray r[3] = {Ray(v0, v1 - v0), Ray(v0, v2 - v0), Ray(v1, v2 - v1)};
    for (int i = 0; i < 3; i++) {
        float tx1 = (this->min.x - r[i].origin.x) * (1.0f / r[i].direction.x);
        float tx2 = (this->max.x - r[i].origin.x) * (1.0f / r[i].direction.x);
        float ty1 = (this->min.y - r[i].origin.y) * (1.0f / r[i].direction.y);
        float ty2 = (this->max.y - r[i].origin.y) * (1.0f / r[i].direction.y);
        float tz1 = (this->min.z - r[i].origin.z) * (1.0f / r[i].direction.z);
        float tz2 = (this->max.z - r[i].origin.z) * (1.0f / r[i].direction.z);

        float tmin = fminf(tx1, tx2);
        float tmax = fmaxf(tx1, tx2);
        tmin = fmaxf(tmin, fminf(ty1, ty2));
        tmax = fminf(tmax, fmaxf(ty1, ty2));
        tmin = fmaxf(tmin, fminf(tz1, tz2));
        tmax = fminf(tmax, fmaxf(tz1, tz2));
        
        // box behind -OR- no i.s -OR- i.s is beyond edge
        if (tmax < 0.0f || tmin > tmax || tmin > 1.0f)
            continue;
    
        return true;
    }
    
    // otherwise not in aabb
    return false;
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