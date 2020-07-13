#ifndef SCENE_H
#define SCENE_H

#include <vector>
#include "geometry.cuh"
#include "hdrloader.h"
#include "math_util.cuh"

class Scene {
private:
    HDRLoaderResult hdri;
    float hdri_exposure = 1.0f;
    float hdri_contrast = 1.0f;
    float hdri_rot_offset = 0.0f;
    bool on_device = false;
    bool has_hdri = false;
    bool use_smooth_hdri = false;
public:
    Scene();
    ~Scene();
    std::vector<Entity*> entities;
    Entity *d_entities;
    size_t n_entities;
    void set_hdri(const std::string &path);
    void set_hdri_exposure(float exposure);
    void set_hdri_contrast(float contrast);
    void rotate_hdri(float amount);
    void add_entity(Entity *entity);
    void copy_to_device();
    void free_from_device();
    void use_hdri_smoothing(bool b);

    __device__ vec3 sample_hdri(vec3 v) const {
        if (!this->has_hdri)
            return vec3(0, 0, 0); 
        int x_i, y_i, pixel_idx;
        float x_c, y_c;
        vec3 e_x(1, 0, 0);
        vec3 e_y(0, 1, 0);
        vec3 e_z(0, 0, 1);
        vec3 v_p = vec3(v.x, 0, v.z).normalized();
        x_c = ((acosf(dot(v_p, e_z)) * (signbit(v_p.x) ?  -1 : 1)) / (2.0f * 3.14159265f)) + 0.5f;
        y_c = fminf(1.0f - ((dot(v, e_y) + 1.0f) / 2.0f), 0.99f); // No, you can't look at the ground. 
        x_c += this->hdri_rot_offset;
        x_c = fmodf(x_c, 1.0f);

        // round down and sample
        if (!this->use_smooth_hdri) {
            x_i = x_c * this->hdri.width;
            y_i = y_c * this->hdri.height;
            pixel_idx = (y_i * this->hdri.width + x_i) * 3; 
            return hdri_exposure * vec3(
                    powf(this->hdri.d_cols[pixel_idx], hdri_contrast),
                    powf(this->hdri.d_cols[pixel_idx + 1], hdri_contrast),
                    powf(this->hdri.d_cols[pixel_idx + 2], hdri_contrast)
            );
        }
        
        // bilinearly interpolate closest pixels and sample
        int x_l = x_c * this->hdri.width;
        int x_r = (x_l + 1) % this->hdri.width;
        int y_l = y_c * this->hdri.height;
        int y_r = (y_l + 1) % this->hdri.height;
        float x_weight = saturate(x_c * this->hdri.width - x_l);
        float y_weight = saturate(y_c * this->hdri.height - y_l);
        int ul_idx = (y_l * this->hdri.width + x_l) * 3;
        int ur_idx = (y_l * this->hdri.width + x_r) * 3;
        int ll_idx = (y_r * this->hdri.width + x_l) * 3;
        int lr_idx = (y_r * this->hdri.width + x_r) * 3;
        vec3 ul(hdri.d_cols[ul_idx], hdri.d_cols[ul_idx + 1], hdri.d_cols[ul_idx + 2]);
        vec3 ur(hdri.d_cols[ur_idx], hdri.d_cols[ur_idx + 1], hdri.d_cols[ur_idx + 2]);
        vec3 ll(hdri.d_cols[ll_idx], hdri.d_cols[ll_idx + 1], hdri.d_cols[ll_idx + 2]);
        vec3 lr(hdri.d_cols[lr_idx], hdri.d_cols[lr_idx + 1], hdri.d_cols[lr_idx + 2]);
        vec3 res = bilerp(ul, ur, ll, lr, x_weight, y_weight);
        res.x = powf(res.x, hdri_contrast);
        res.y = powf(res.y, hdri_contrast);
        res.z = powf(res.z, hdri_contrast);
        return hdri_exposure * res;
    }
};

#endif
