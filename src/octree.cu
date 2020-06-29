#include "geometry.cuh"
#include "util.cuh"

void Octree::pretty_print(int child_nr) {
    for(int i = 0; i < this->depth; i++) printf("  ");
    printf("[%d] : ", child_nr);
    for(size_t i = 0; i < this->triangle_indices.size(); i++) printf("%d, ", this->triangle_indices[i]);
    printf("\n");
    for(int i = 0; i < 8; i++) {
        if(this->children[i] != nullptr){
            this->children[i]->pretty_print(i);
        }
    }
}

void Octree::copy_to_device() {
    // copy triangle indices
    long size = this->triangle_indices.size()*sizeof(int);
    gpuErrchk(cudaMalloc(&this->d_triangle_indices, size));
    gpuErrchk(cudaMemcpy(this->d_triangle_indices, &(this->triangle_indices[0]), size, cudaMemcpyHostToDevice));

    // copy children
    for (int i = 0; i < 8; i++) {
        if (this->children[i] == nullptr)
            continue;
        this->children[i]->copy_to_device();
        long size = sizeof(Octree);
        gpuErrchk(cudaMalloc(&(this->d_children[i]), size));
        gpuErrchk(cudaMemcpy(this->d_children[i], this->children[i], size, cudaMemcpyHostToDevice));
        //gpuErrchk(cudaPeekAtLastError());
    }

    this->on_device = true;
}

void Octree::free_from_device() {
    if (!this->on_device)
        return;
        
    gpuErrchk(cudaFree(this->d_triangle_indices));

    // free children
    for (int i = 0; i < 8; i++) {
        if (this->children[i] == nullptr)
            continue;
        this->children[i]->free_from_device();
        gpuErrchk(cudaFree(this->d_children[i])); // this should still work
    }

    this->on_device = false;
}

Octree::~Octree() {
    if (this->on_device) {
        std::cout << "Octree still on device! :(" << std::endl;
    }

    // free children
    for (int i = 0; i < 8; i++) {
        if (this->children[i] == nullptr)
            continue;
        delete this->children[i];
    }
}

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
    //int _case = -1; // 0 no single child fits triangle

    AABB c[8];
    c[0] = AABB(vec3(x_min, y_min, z_min), vec3(x_mid, y_mid, z_mid)); // old --> 0
    c[4] = AABB(vec3(x_mid, y_min, z_min), vec3(x_max, y_mid, z_mid)); // old --> 1
    c[1] = AABB(vec3(x_min, y_min, z_mid), vec3(x_mid, y_mid, z_max)); // old --> 2
    c[5] = AABB(vec3(x_mid, y_min, z_mid), vec3(x_max, y_mid, z_max)); // old --> 3
    c[2] = AABB(vec3(x_min, y_mid, z_min), vec3(x_mid, y_max, z_mid)); // old --> 4
    c[6] = AABB(vec3(x_mid, y_mid, z_min), vec3(x_max, y_max, z_mid)); // old --> 5
    c[3] = AABB(vec3(x_min, y_mid, z_mid), vec3(x_mid, y_max, z_max)); // old --> 6
    c[7] = AABB(vec3(x_mid, y_mid, z_mid), vec3(x_max, y_max, z_max)); // old --> 7

    // old way
    /*for (int i = 0; i < 8; i++) {
        _case = (c[i].contains_triangle(v0, v1, v2)) ? i : _case;
    }

    // Unless the math is wrong, there's exactly one possible case (no overridden values)
    if (_case == -1 || this->depth == MAX_OCTREE_DEPTH) {
        this->triangle_indices.push_back(triangle_idx);
        this->n_triangle_indices++;
    } else {
        if (this->children[_case] == nullptr) {
            this->children[_case] = new Octree(c[_case], this->depth + 1);
        }
        this->children[_case]->insert_triangle(v0, v1, v2, triangle_idx);
    }*/

    // new way
    if (this->depth == MAX_OCTREE_DEPTH) {
        // If this is a leaf add triangle
        this->triangle_indices.push_back(triangle_idx);
        this->n_triangle_indices++;
    } else {
        // Else add to children
        for (int i = 0; i < 8; i++) {
            if (!c[i].intersects_triangle(v0, v1, v2))
                continue;

            if (this->children[i] == nullptr) {
                this->children[i] = new Octree(c[i], this->depth + 1);
            }

            this->children[i]->insert_triangle(v0, v1, v2, triangle_idx);
        }
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