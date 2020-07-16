#include "post_processing.cuh"
#include <assert.h>
#include <cmath>
#include <vector>
#include "util.cuh"

/*
 * Kernel
 */
void Kernel::print() {
    // we copy from device memory every time. But it's fine. print() is only for debugging.
    size_t buf_size = (this->size*2+1) * (this->size*2+1);
    float *temp_buf = new float[buf_size];
    gpuErrchk(cudaMemcpy(temp_buf, this->d_buf, buf_size*sizeof(float), cudaMemcpyDeviceToHost));
    float sum = 0.0f;
    for (size_t y = 0; y < size*2 + 1; y++) {
        for (size_t x = 0; x < size*2 + 1; x++) {
            printf("%d ", int(10 * temp_buf[y * (2*size + 1) + x])); 
            sum += temp_buf[y * (2*size + 1) + x];
        }
        printf("\n");
    }
    printf("sum: %g\n", sum);
    delete [] temp_buf;
}

float sample_gaussian(float x, float y, float sigma) {
    // @Incomplete maybe integrate from (x,y) - 0.5 to (x,y) + 0.5
    // for more accurate samples
    float fac = 1.0f / (sigma*sigma*2*M_PI);
    float exp = -((x*x+y*y) / (2*sigma*sigma));
    return fac * powf(M_E, exp);
}

/*
 * Makes a gaussian kernel. Distribution is normalized to the kernel size.
 * All kernel samples are within [(-1,-1), (1, 1)] of a gaussian distribution.
 */
void Kernel::make_gaussian(unsigned int _size, float sigma) {
    assert(this->d_buf == nullptr);
    this->size = _size;
    size_t buf_size = (2 * size + 1) * (2 * size + 1);
    float *temp_buf = new float[buf_size];
    int ssize = _size;
    for (int y = 0; y < ssize*2 + 1; y++) {
        for (int x = 0; x < ssize*2 + 1; x++) {
            size_t idx = y * (this->size * 2  + 1) + x;
            float sample_pos_x = float(x - ssize) / float(ssize);
            float sample_pos_y = float(y - ssize) / float(ssize);
            temp_buf[idx] = sample_gaussian(sample_pos_x, sample_pos_y, sigma);
        }
    } 
    gpuErrchk(cudaMalloc(&(this->d_buf), buf_size * sizeof(float)));
    gpuErrchk(cudaMemcpy(this->d_buf, temp_buf, buf_size*sizeof(float), cudaMemcpyHostToDevice));
    delete[] temp_buf;
}

void Kernel::make_mean(unsigned int _size) {
    assert(this->d_buf == nullptr);
    this->size = _size;
    size_t buf_size = (2 * size + 1) * (2 * size + 1);
    float *temp_buf = new float[buf_size];
    for (size_t i = 0; i < buf_size; i++) {
        temp_buf[i] = 1.0f;
    }
    gpuErrchk(cudaMalloc(&(this->d_buf), buf_size * sizeof(float)));
    gpuErrchk(cudaMemcpy(this->d_buf, temp_buf, buf_size*sizeof(float), cudaMemcpyHostToDevice));
    delete[] temp_buf;
}

__device__
float Kernel::at(int x, int y) const {
    y = this->size + y;
    x = this->size + x;
    size_t idx = y * (this->size * 2 + 1) + x;
    return this->d_buf[idx];
}


void apply_filter(std::vector<vec3> &buf, ivec2 resolution, const Kernel &kernel) {
    // prepare buffers and kernel
    size_t buf_size = resolution.x * resolution.y * sizeof(vec3);
    vec3 *in_buf;
    vec3 *out_buf;
    Kernel *d_kernel;
    gpuErrchk(cudaMalloc(&in_buf, buf_size));
    gpuErrchk(cudaMalloc(&out_buf, buf_size));
    gpuErrchk(cudaMalloc(&d_kernel, sizeof(Kernel)));
    gpuErrchk(cudaMemcpy(in_buf, &(buf[0]), buf_size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_kernel, &kernel, sizeof(Kernel), cudaMemcpyHostToDevice));

    // prepare blocks & threads
    int tile_size = 8;
    dim3 blocks(
            resolution.x / tile_size, 
            resolution.y / tile_size
    );
    dim3 threads(tile_size, tile_size);

    // apply filter on image on device 
    device_apply_filter<<<blocks, threads>>>(out_buf, in_buf, buf_size, resolution, d_kernel);
    gpuErrchk(cudaDeviceSynchronize());

    // copy results
    gpuErrchk(cudaMemcpy(&(buf[0]), out_buf, buf_size, cudaMemcpyDeviceToHost)); 

    // cleanup
    gpuErrchk(cudaFree(in_buf));
    gpuErrchk(cudaFree(out_buf));
    gpuErrchk(cudaFree(d_kernel));
}


__global__ 
void device_apply_filter(vec3 *out_buf, vec3 *in_buf, size_t buf_size, ivec2 resolution, Kernel *kernel) {
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    int pixel_idx = v * resolution.x +  u;

    int width  = resolution.x;
    int height = resolution.y;
    int ks     = kernel->size;
    
    float kernel_weight = 0.0f;
    vec3 pixel(0.0f); 
    for (int y = -ks; y <= ks; y++) {
        for (int x = -ks; x <= ks; x++) {
            int img_y = v + y;
            int img_x = u + x;
            size_t img_idx = img_y * width + img_x;
            if (img_y < 0 || img_y >= height || img_x < 0 || img_x >= width)
                continue; // out of image bounds (crop for now)
            pixel = pixel + kernel->at(x, y) * in_buf[img_idx]; 
            kernel_weight += kernel->at(x, y);
        }
    }
    pixel = pixel / kernel_weight; 
    out_buf[pixel_idx] = pixel;
}

/*
 * Post Processing
 */
vec3 ACESFilm(vec3 x)
{
    // From: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
    float a = 2.51f;
    float b = 0.03f;
    float c = 2.43f;
    float d = 0.59f;
    float e = 0.14f;
    vec3 res = (x*(a*x+b))/(x*(c*x+d)+e);
    res.x = max(0.0f, min(1.0f, res.x));
    res.y = max(0.0f, min(1.0f, res.y));
    res.z = max(0.0f, min(1.0f, res.z));
    return res;
}

void normalize_image(
        std::vector<vec3> &buf, 
        unsigned int n_samples_per_pixel
) {
    for (size_t i = 0; i < buf.size(); i++) {
        buf[i] = buf[i] / n_samples_per_pixel;
    }
}

void apply_aces(std::vector<vec3> &buf) {
    for (size_t i = 0; i < buf.size(); i++) {
        buf[i] = ACESFilm(buf[i]);
    }
}

void gamma_correct(std::vector<vec3> &buf, float gamma) {
    for (size_t i = 0; i < buf.size(); i++) {
        buf[i].x = pow(buf[i].x, 1.0f / gamma);
        buf[i].y = pow(buf[i].y, 1.0f / gamma);
        buf[i].z = pow(buf[i].z, 1.0f / gamma);
    }
}

void compound_buffers(
        std::vector<vec3> &out_image,
        const std::vector<vec3> &in_buf, 
        unsigned int n_split_buffers
) {
    std::cout << "capacity: " << out_image.capacity() << std::endl;
    size_t single_buffer_size = out_image.capacity();
    for (size_t i = 0; i < single_buffer_size; i++) {
        vec3 sum(0);
        for(unsigned int j = 0; j < n_split_buffers; j++) {
            sum = sum + in_buf[i + j * single_buffer_size];
        }
        out_image[i] = sum;
    }
}

inline float luminosity(vec3 color) {
    return dot(color, vec3(0.21f, 0.72f, 0.07f));
}

std::vector<vec3> apply_threshold(const std::vector<vec3> &in_buf, float threshold) {
    std::vector<vec3> result(in_buf.size());
    for (size_t i = 0; i < in_buf.size(); i++) {
        float l = luminosity(in_buf[i]);
        result[i] = (l < threshold) ? vec3(0.0f) : in_buf[i]; 
    }
    return result;
}

void image_add(std::vector<vec3> &buf, const std::vector<vec3> &layer, float opacity) {
    assert(buf.size() == layer.size()); 
    for (size_t i = 0; i < buf.size(); i++) {
        buf[i] = buf[i] + opacity * layer[i];
    }
}

