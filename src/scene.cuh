#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include "geometry.cuh"
#include "hdrloader.h"

class Scene {
private:
    HDRLoaderResult hdri;
    bool on_device = false;
    bool has_hdri = false;
public:
    Scene();
    ~Scene();
    std::vector<Entity*> entities;
    Entity *d_entities;
    void set_hdri(const std::string &path);
    void add_entity(Entity *entity);
    void copy_to_device();
    void free_from_device();

    __host__ __device__ vec3 sample_hdri(const vec3 &v) const {
        if (!this->has_hdri)
            return vec3(0, 0, 0); 
        int x_i, y_i, pixel_idx;
        float x_c, y_c;
        vec3 e_x(1, 0, 0);
        vec3 e_y(0, 1, 0);
        vec3 e_z(0, 0, 1);
        vec3 v_p = vec3(v.x, 0, v.z).normalized();
        x_c = ((acosf(dot(v_p, e_z)) * (signbit(v_p.x) ?  -1 : 1)) / (2.0f * 3.14159265f)) + 0.5f;
        y_c = 1.0f - ((dot(v, e_y) + 1.0f) / 2.0f);
        x_i = x_c * this->hdri.width;
        y_i = y_c * this->hdri.height;
        pixel_idx = (y_i * this->hdri.width + x_i) * 3; 
#ifndef __CUDA_ARCH__
        return vec3(
                this->hdri.cols[pixel_idx],
                this->hdri.cols[pixel_idx + 1],
                this->hdri.cols[pixel_idx + 2]
        );
#else
        return vec3(
                this->hdri.d_cols[pixel_idx],
                this->hdri.d_cols[pixel_idx + 1],
                this->hdri.d_cols[pixel_idx + 2]
        );
#endif
    }
};

#endif