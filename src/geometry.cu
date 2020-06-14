#define MAX_OCTREE_DEPTH 4

#include "geometry.cuh"
#include <math.h>
#include <stdlib.h>
#include <iostream>
#include <vector>
#include "util.cuh"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

/************************************************************************************/
/*                                    Constructors                                  */
/************************************************************************************/

Entity::Entity(const std::string &path, const Material &material) {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;

    bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,  path.c_str());

    if(!warn.empty()) {
        std::cout << warn << std::endl;
    }

    if(!err.empty()) {
        std::cout << err << std::endl;
    }

    if (!ret || shapes.size() != 1) {
        exit(1);
    }

    bool no_normals = (attrib.normals.size() / 3) == 0;

    // allocate space
    this->n_vertices  = attrib.vertices.size() / 3;
    this->n_triangles = shapes[0].mesh.num_face_vertices.size();
    this->vertices  = new Vertex[this->n_vertices];
    this->triangles = new Triangle[this->n_triangles];
    
    // For AABB
    vec3 min(FLT_MAX, FLT_MAX, FLT_MAX);
    vec3 max(-FLT_MAX, -FLT_MAX, -FLT_MAX);

    // load vertices
    size_t index_offset = 0;
    for (size_t v = 0; v < this->n_vertices; v++) {
        vec3 position(
                (float) attrib.vertices[index_offset + 0],
                (float) attrib.vertices[index_offset + 1],
                (float) attrib.vertices[index_offset + 2]
        );

        vec3 normal(0.0f, 0.0f, 0.0f);
        if(!no_normals) {
            normal.x = attrib.normals[index_offset + 0];
            normal.y = attrib.normals[index_offset + 1];
            normal.z = attrib.normals[index_offset + 2];
            normal.normalize();
        }

        // update min, max for aabb
        min.x = (position.x < min.x) ? position.x : min.x;
        min.y = (position.y < min.y) ? position.y : min.y;
        min.z = (position.z < min.z) ? position.z : min.z;
        max.x = (position.x > max.x) ? position.x : max.x;
        max.y = (position.y > max.y) ? position.y : max.y;
        max.z = (position.z > max.z) ? position.z : max.z;

        this->vertices[v] = Vertex(position, normal);
        index_offset += 3;
    }

    // load triangles
    tinyobj::shape_t shape = shapes[0];
    index_offset = 0;
    for (size_t f = 0; f < this->n_triangles; f++) {
        if ((int)shape.mesh.num_face_vertices[f] != 3) {
            std::cerr << "OBJ file faces must be triangles" << std::endl;
            exit(1);
        }

        this->triangles[f] = Triangle(
                shape.mesh.indices[index_offset + 0].vertex_index,
                shape.mesh.indices[index_offset + 1].vertex_index,
                shape.mesh.indices[index_offset + 2].vertex_index
        );

        if (no_normals) {
            int v0_idx = this->triangles[f].idx_a;
            int v1_idx = this->triangles[f].idx_b;
            int v2_idx = this->triangles[f].idx_c;
            vec3 v0 = this->vertices[v0_idx].position;
            vec3 v1 = this->vertices[v1_idx].position;
            vec3 v2 = this->vertices[v2_idx].position;
            vec3 e1 = v1 - v0;
            vec3 e2 = v2 - v0;
            vec3 t_normal = cross(e1, e2).normalized();
            this->vertices[v0_idx].normal = this->vertices[v0_idx].normal + t_normal;
            this->vertices[v1_idx].normal = this->vertices[v1_idx].normal + t_normal;
            this->vertices[v2_idx].normal = this->vertices[v2_idx].normal + t_normal;
        }

        index_offset += 3;
    }

    // normalize normals
    if (no_normals) {
        for (size_t i = 0; i < this->n_vertices; i++) {
            this->vertices[i].normal.normalize();
        }
    }

    // create AABB
    this->aabb = AABB(min, max);

    // set center
    this->center = vec3(
            (min.x + max.x) / 2,
            (min.y + max.y) / 2,
            (min.z + max.z) / 2
    );

    // set shape & material
    this->material = material;
    this->shape = TRIANGLE_MESH;
}

Entity::Entity(const vec3 &center, float radius, const Material &material) {
    this->shape     = SPHERE;
    this->center    = center;
    this->radius    = radius;
    this->material  = material;
}

/************************************************************************************/
/*                                   Octree stuff                                   */
/************************************************************************************/

void Octree::insert_triangle(vec3 v0, vec3 v1, vec3 v2, size_t triangle_idx) {
    float x_min = this->region.min.x;
    float y_min = this->region.min.y;
    float z_min = this->region.min.z;
    float x_max = this->region.max.x;
    float y_max = this->region.max.y;
    float z_max = this->region.max.z;
    float x_mid = 0.5f * (x_min + x_max);
    float y_mid = 0.5f * (y_min + y_max);
    float z_mid = 0.5f * (z_min + z_max);
    int _case = -1; // 0 no single child fits triangle
    AABB c[8];

    ////     Bottom: y_min = 0      Top: y_min = 1/2
    ////      _____________         _____________
    ////     |      |      |       |      |      |        ^ Z
    ////     |  c2  |  c3  |       |  c6  |  c7  |        |
    ////     |______|______|       |______|______|        |
    ////     |      |      |       |      |      |        |
    ////     |  c0  |  c1  |       |  c4  |  c5  |          ------> X
    ////     |______|______|       |______|______|
    ////
    ////             <----- lower ----->        <----- upper -----> 
    c[0] = AABB(vec3(x_min, y_min, z_min), vec3(x_mid, y_mid, z_mid));
    c[1] = AABB(vec3(x_mid, y_min, z_min), vec3(x_max, y_mid, z_mid));
    c[2] = AABB(vec3(x_min, y_min, z_mid), vec3(x_mid, y_mid, z_max));
    c[3] = AABB(vec3(x_mid, y_min, z_mid), vec3(x_max, y_mid, z_max));
    c[4] = AABB(vec3(x_min, y_mid, z_min), vec3(x_mid, y_max, z_mid));
    c[5] = AABB(vec3(x_mid, y_mid, z_min), vec3(x_max, y_max, z_mid));
    c[6] = AABB(vec3(x_min, y_mid, z_mid), vec3(x_mid, y_max, z_max));
    c[7] = AABB(vec3(x_mid, y_mid, z_mid), vec3(x_max, y_max, z_max));
    for (int i = 0; i < 8; i++) {
        _case = (c[i].contains_triangle(v0, v1, v2)) ? i : _case;
    }

    // Unless the math is wrong, there's exactly one possible case (no overridden values)
    if (_case == -1 || this->depth == MAX_OCTREE_DEPTH) {
        std::cout << "asd" << std::endl;
        std::cout << this->triangle_indices.size() << std::endl;
        this->triangle_indices.push_back(triangle_idx);
    } else {
        if (this->children[_case] == nullptr) {
            std::cout << "af" << std::endl;
            this->children[_case] = new Octree(c[_case], this->depth + 1);
        }

        this->children[_case]->insert_triangle(v0, v1, v2, triangle_idx);
    }
}

void Octree::insert_triangles(Vertex *vertices, Triangle *triangles, size_t n_triangles) {
    for (size_t i = 0; i < n_triangles; i++) {
        this->insert_triangle(
                vertices[triangles[i].idx_a].position,
                vertices[triangles[i].idx_b].position,
                vertices[triangles[i].idx_c].position,
                i
        );
    }
}

/************************************************************************************/
/*                                 Misc. functions                                  */
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

    // a = alpha, b = beta, g = gamma. For rotation on z, y and x respectively.
    float cos_a = cos(rot.z);
    float sin_a = sin(rot.z);
    float cos_b = cos(rot.y);
    float sin_b = sin(rot.y);
    float cos_g = cos(rot.x);
    float sin_g = sin(rot.x);

    // Rotation matrix R:
    vec3 R0(/*[0,0]*/ cos_a * cos_b, /*[1,0]*/  cos_a*sin_b*sin_g - sin_a*cos_g, /*[2,0]*/  cos_a*sin_b*cos_g + sin_a*sin_g);
    vec3 R1(/*[0,1]*/ sin_a * cos_b, /*[1,1]*/  sin_a*sin_b*sin_g + cos_a*cos_g, /*[2,1]*/  sin_a*sin_b*cos_g - cos_a*sin_g);
    vec3 R2(/*[0,2]*/ -sin_b,        /*[1,2]*/  cos_b*sin_g,                     /*[2,2]*/  cos_b*cos_g);
    
    // rotate on x
    for (size_t i = 0; i < this->n_vertices; i++) {
        // rotate vertex positions
        vec3 v = this->vertices[i].position - this->center;
        v = vec3(dot(v, R0), dot(v, R1), dot(v, R2));
        this->vertices[i].position = v + this->center;

        // rotate vertex normals
        vec3 n = this->vertices[i].normal;
        n = vec3(dot(n, R0), dot(n, R1), dot(n, R2));
        this->vertices[i].normal = n;
    }

    // recalculate aabb
    this->aabb.recalculate(this->vertices, this->n_vertices);
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