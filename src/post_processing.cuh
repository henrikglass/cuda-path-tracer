#ifndef POST_PROCESSING_H
#define POST_PROCESSING_H

#include "vector.cuh"
#include <vector>
#include "util.cuh"

struct Kernel {
    float *d_buf = nullptr;
    unsigned int size; // 3x3 kernel has size = 1, 5x5 has size = 2
    float weight;
    Kernel(){}
    ~Kernel(){
        if (d_buf != nullptr) {
            gpuErrchk(cudaFree(d_buf)); 
            d_buf = nullptr;
        }
    }
    void print();
    void make_gaussian(unsigned int _size, float sigma);
    void make_mean(unsigned int _size);
    __device__ float at(int y, int x) const;
    //void make_from_image(...) // @Incomplete implement this later for pretty bloom effects
};


void compound_buffers(std::vector<vec3> &out_image, const std::vector<vec3> &in_buf, unsigned int n_split_buffers);
void normalize_image(std::vector<vec3> &buf, unsigned int n_samples_per_pixel);
void gamma_correct(std::vector<vec3> &buf, float gamma);
void apply_aces(std::vector<vec3> &buf);
void image_add(std::vector<vec3> &buf, const std::vector<vec3> &layer);

std::vector<vec3> apply_threshold(const std::vector<vec3> &in_buf, float threshold);
void apply_filter(std::vector<vec3> &buf, ivec2 resolution, const Kernel &kernel);

__global__ void device_apply_filter(vec3 *out_buf, vec3 *in_buf, size_t buf_size, ivec2 resolution, Kernel *kernel);

#endif
