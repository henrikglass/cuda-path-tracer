#include "geometry.cuh"
#include "util.cuh"

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

/*
 * Triangle mesh.
 */
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
    tinyobj::shape_t _shape = shapes[0];
    index_offset = 0;
    for (size_t f = 0; f < this->n_triangles; f++) {
        if ((int)_shape.mesh.num_face_vertices[f] != 3) {
            std::cerr << "OBJ file faces must be triangles" << std::endl;
            exit(1);
        }

        this->triangles[f] = Triangle(
                _shape.mesh.indices[index_offset + 0].vertex_index,
                _shape.mesh.indices[index_offset + 1].vertex_index,
                _shape.mesh.indices[index_offset + 2].vertex_index
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

/*
 * Sphere.
 */
Entity::Entity(const vec3 &center, float radius, const Material &material) {
    this->shape     = SPHERE;
    this->center    = center;
    this->radius    = radius;
    this->material  = material;
}

void Entity::construct_octree() {
    this->octree = new Octree(this->aabb, 0);
    this->octree->insert_triangles(this->vertices, this->triangles, this->n_triangles);
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

        // copy octree
        // TODO nullcheck for entities that don't have octrees
        if (this->octree != nullptr) {
            this->octree->copy_to_device();
            gpuErrchk(cudaMalloc(&this->d_octree, sizeof(Octree)));
            gpuErrchk(cudaMemcpy(this->d_octree, this->octree, sizeof(Octree), cudaMemcpyHostToDevice));
            //gpuErrchk(cudaPeekAtLastError());
        }
    }

    this->on_device = true;
}

void Entity::free_from_device() {
    if (this->shape == SPHERE)
        return;

    // mesh case
    if (this->d_octree != nullptr) {
        this->octree->free_from_device();
        gpuErrchk(cudaFree(this->d_octree));
    }

    if (this->d_vertices != nullptr) {
        gpuErrchk(cudaFree(this->d_vertices));
    }

    if (this->d_triangles != nullptr) {
        gpuErrchk(cudaFree(this->d_triangles));
    }

    this->on_device = false;
}

/*
 * Destructor. Must be called after free_from_device().
 */
Entity::~Entity(){
    if (this->on_device){
        std::cout << "entity still on device!! :(" << std::endl;
        exit(1);
        return;
    }

    if (this->octree != nullptr) {
        delete this->octree;
    }

    if (this->vertices != nullptr) {
        delete[] this->vertices;
    }

    if (this->triangles != nullptr) {
        delete[] this->triangles;
    }
}