#include "geometry.cuh"
#include <math.h>

void AABB::recalculate(Vertex *vertices, int n_vertices) {
    vec3 _min(FLT_MAX, FLT_MAX, FLT_MAX);
    vec3 _max(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for (int i = 0; i < n_vertices; i++) {
        vec3 position = vertices[i].position;
        _min.x = (position.x < _min.x) ? position.x : _min.x;
        _min.y = (position.y < _min.y) ? position.y : _min.y;
        _min.z = (position.z < _min.z) ? position.z : _min.z;
        _max.x = (position.x > _max.x) ? position.x : _max.x;
        _max.y = (position.y > _max.y) ? position.y : _max.y;
        _max.z = (position.z > _max.z) ? position.z : _max.z;
    }
    
    this->min = _min - vec3(AABB_PADDING, AABB_PADDING, AABB_PADDING); // padding
    this->max = _max + vec3(AABB_PADDING, AABB_PADDING, AABB_PADDING);
}

__host__ 
bool AABB::contains_triangle(vec3 v0, vec3 v1, vec3 v2) {
    return triangle_inside_aabb(
            this->min.x, this->min.y, this->min.z, 
            this->max.x, this->max.y, this->max.z,
            v0, v1, v2
    );
}

int sign(float f) {
    return (f < 0.0f) ? -1 : 1;
}

bool plane_aabb_overlap(const vec3 &normal, float d, const vec3 &max_box) {
    vec3 v_min, v_max;
    v_min.x = -sign(normal.x) * max_box.x;
    v_max.x =  sign(normal.x) * max_box.x;
    v_min.y = -sign(normal.y) * max_box.y;
    v_max.y =  sign(normal.y) * max_box.y;
    v_min.z = -sign(normal.z) * max_box.z;
    v_max.z =  sign(normal.z) * max_box.z;
    if ((dot(normal, v_min) + d) > 0.0f) return false;
    if ((dot(normal, v_max) + d) >= 0.0f) return true;
    return false;
}

/*
 * @Incomplete Missing a SAT test. But seems to work anyways.
 * Otherwise add the extra SAT tests:  
 * https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/pubs/tribox.pdf
 */
 __host__ 
bool AABB::intersects_triangle(const vec3 &u0, const vec3 &u1, const vec3 &u2) {
    // aabb to center-extents representation
    vec3 aabb_center = 0.5f * (this->min + this->max);
    vec3 extents = (this->max - this->min) / 2;

    // triangle vertices
    vec3 v[3] = {u0 - aabb_center, u1 - aabb_center, u2 - aabb_center};

    // triangle edges
    vec3 f[3] = {v[1] - v[0], v[2] - v[1], v[0] - v[2]};

    // xyz axes @Incomplete constexpr?
    vec3 e[3] = {vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f)};

    // compute the 9 triangle edge axes
    vec3 a[3][3];
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            a[i][j] = cross(e[i], f[j]);
        }
    }

    // 9 SAT tests for triangle edge axes
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
            float p0 = dot(v[0], a[i][j]);
            float p1 = dot(v[1], a[i][j]);
            float p2 = dot(v[2], a[i][j]);
            float r  = extents.x * fabs(dot(e[0], a[i][j])) +
                    extents.y * fabs(dot(e[1], a[i][j])) +
                    extents.z * fabs(dot(e[2], a[i][j]));
            float p_min = fmin(p0, fmin(p1, p2));
            float p_max = fmax(p0, fmax(p1, p2));
            if (fmax(-p_max, p_min) > r)
                return false;
        }
    }

    // 3 SAT tests for aabb
    float min_x = fmin(v[0].x, fmin(v[1].x, v[2].x));
    float min_y = fmin(v[0].y, fmin(v[1].y, v[2].y));
    float min_z = fmin(v[0].z, fmin(v[1].z, v[2].z));
    float max_x = fmax(v[0].x, fmax(v[1].x, v[2].x));
    float max_y = fmax(v[0].y, fmax(v[1].y, v[2].y));
    float max_z = fmax(v[0].z, fmax(v[1].z, v[2].z));
    if (min_x > extents.x || max_x < -extents.x) return false;
    if (min_y > extents.y || max_y < -extents.y) return false;
    if (min_z > extents.z || max_z < -extents.z) return false;

    // test if triangle plane intersects @Incomplete generates false negatives for some reason.
    //vec3 normal = cross(e[0], e[1]);
    //float d = -dot(normal, v[0]);
    //if (!plane_aabb_overlap(normal, d, extents)) return false;

    // all tests passed
    return true;
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