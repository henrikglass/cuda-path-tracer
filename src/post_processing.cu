#include "post_processing.cuh"
#include <assert.h>
#include <cmath>
#include <vector>
#include "util.cuh"
#include "math_util.cuh"

/**
 * Debug: Prints contents of kernel
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

/**
 * Samples a 1D gaussian function.
 */
__device__ float sample_gaussian_1d(float x, float sigma) {
    float fac = 1.0f / (sigma*sigma*2*C_PI);
    float exp = -((x*x) / (2*sigma*sigma));
    return fac * powf(C_E, exp);
}

/**
 * Samples a 2D gaussian function.
 */
float sample_gaussian_2d(float x, float y, float sigma) {
    // @Incomplete maybe integrate from (x,y) - 0.5 to (x,y) + 0.5
    // for more accurate samples
    float fac = 1.0f / (sigma*sigma*2*M_PI);
    float exp = -((x*x+y*y) / (2*sigma*sigma));
    return fac * powf(M_E, exp);
}

/**
 * Makes a gaussian kernel.
 */
void Kernel::make_gaussian(unsigned int _size, float sigma) {
    assert(this->d_buf == nullptr);
    this->size      = _size;
    size_t buf_size = (2 * size + 1) * (2 * size + 1);
    float *temp_buf = new float[buf_size];
    int ssize       = _size;
    for (int y = 0; y < ssize*2 + 1; y++) {
        for (int x = 0; x < ssize*2 + 1; x++) {
            size_t idx = y * (this->size * 2  + 1) + x;
            float sample_pos_x = float(x - ssize);
            float sample_pos_y = float(y - ssize);
            temp_buf[idx] = sample_gaussian_2d(sample_pos_x, sample_pos_y, sigma);
        }
    } 
    gpuErrchk(cudaMalloc(&(this->d_buf), buf_size * sizeof(float)));
    gpuErrchk(cudaMemcpy(this->d_buf, temp_buf, buf_size*sizeof(float), cudaMemcpyHostToDevice));
    delete[] temp_buf;
}

/**
 * Makes a uniform distribution kernel.
 */
void Kernel::make_mean(unsigned int _size) {
    assert(this->d_buf == nullptr);
    this->size      = _size;
    size_t buf_size = (2 * size + 1) * (2 * size + 1);
    float *temp_buf = new float[buf_size];
    for (size_t i = 0; i < buf_size; i++) {
        temp_buf[i] = 1.0f;
    }
    gpuErrchk(cudaMalloc(&(this->d_buf), buf_size * sizeof(float)));
    gpuErrchk(cudaMemcpy(this->d_buf, temp_buf, buf_size*sizeof(float), cudaMemcpyHostToDevice));
    delete[] temp_buf;
}

/**
 * Samples kernel (x,y), where (0,0) is the center. 
 */
__device__
float Kernel::at(int x, int y) const {
    y = this->size + y;
    x = this->size + x;
    size_t idx = y * (this->size * 2 + 1) + x;
    return this->d_buf[idx];
}

/**
 * Wrapper function for applying a kernel (filter) to an image buffer on the device.
 * 
 * @param buf           the image buffer
 * @param resolution    the image resolution
 * @param kernel        the kernel (filter) to be applied
 * @param filter_type   specifies the filter type. Default is `NORMAL`
 */
void apply_filter(
        std::vector<vec3> &buf,
        ivec2 resolution,
        const Kernel &kernel,
        FilterType filter_type
) {
    // prepare buffers and kernel
    size_t buf_size  = buf.size() * sizeof(vec3);
    vec3 *in_buf     = prepare_cuda_buffer(buf);
    vec3 *out_buf    = prepare_empty_cuda_buffer<vec3>(buf.size());
    Kernel *d_kernel = prepare_cuda_instance(kernel);
    int tile_size    = 8;

    // apply filter on image on device 
    dim3 blocks(resolution.x / tile_size, resolution.y / tile_size);
    dim3 threads(tile_size, tile_size);
    device_apply_filter<<<blocks, threads>>>(out_buf, in_buf, buf_size, resolution, d_kernel, filter_type);
    gpuErrchk(cudaDeviceSynchronize());
    
    // copy results
    gpuErrchk(cudaMemcpy(&(buf[0]), out_buf, buf_size, cudaMemcpyDeviceToHost)); 
    
    // cleanup
    gpuErrchk(cudaFree(in_buf));
    gpuErrchk(cudaFree(out_buf));
    gpuErrchk(cudaFree(d_kernel));
}

/**
 * Applies a kernel to an image buffer on the device.
 * 
 * @param out_buf       the output image buffer
 * @param in_buf        the input image buffer
 * @param buf_size      the size of `out_buf` and `in_buf` in bytes
 * @param resolution    the image resolution
 * @param kernel        the kernel (filter) to be applied
 * @param filter_type   specifies the filter type. Default is `NORMAL`
 */
__global__ 
void device_apply_filter(
        vec3 *out_buf,
        vec3 *in_buf, 
        size_t buf_size,
        ivec2 resolution,
        Kernel *kernel,
        FilterType filter_type
) {
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
            int neighbour_y = v + y;
            int neighbour_x = u + x;
            size_t neighbour_idx = neighbour_y * width + neighbour_x;

            // out of image bounds (crop for now)
            if (neighbour_y < 0 || neighbour_y >= height || 
                    neighbour_x < 0 || neighbour_x >= width)
                continue; 

            float gs, gi, w;
            gs = kernel->at(x, y);
            //@Incomplete Maybe move this to outside the nested for loops in case
            // the compiler can't manage it by itself. This is prettier though.
            switch (filter_type) {
                case NORMAL: gi = 1.0f; break;
                case BILATERAL: gi = sample_gaussian_1d(255*luminosity(in_buf[neighbour_idx] - in_buf[pixel_idx]), 24.0f); break;
            }
            w = gi * gs;
            pixel = pixel + w * in_buf[neighbour_idx]; 
            kernel_weight += w;
        }
    }
    pixel = pixel / kernel_weight; 
    out_buf[pixel_idx] = pixel;
}

/**
 * Applies ACES-style color correction to a single color value.
 * 
 * From: https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
 */
vec3 ACESFilm(vec3 x) {
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

/**
 * Normalizes the pixel color values in `buf` by the total number of
 * samples per pixel.
 */
void normalize_image(
        std::vector<vec3> &buf, 
        unsigned int n_samples_per_pixel
) {
    for (size_t i = 0; i < buf.size(); i++) {
        buf[i] = buf[i] / n_samples_per_pixel;
    }
}

/**
 * Applies ACES-style color correction to an image buffer.
 */
void apply_aces(std::vector<vec3> &buf) {
    for (size_t i = 0; i < buf.size(); i++) {
        buf[i] = ACESFilm(buf[i]);
    }
}

/**
 * Applies gamma correction to an image buffer.
 */
void gamma_correct(std::vector<vec3> &buf, float gamma) {
    for (size_t i = 0; i < buf.size(); i++) {
        buf[i].x = pow(buf[i].x, 1.0f / gamma);
        buf[i].y = pow(buf[i].y, 1.0f / gamma);
        buf[i].z = pow(buf[i].z, 1.0f / gamma);
    }
}

/**
 * Compounds (adds) multiple image buffers together.
 */
void compound_buffers(
        std::vector<vec3> &out_image,
        const std::vector<vec3> &in_buf, 
        unsigned int n_split_buffers
) {
    size_t single_buffer_size = out_image.capacity();
    for (size_t i = 0; i < single_buffer_size; i++) {
        vec3 sum(0);
        for(unsigned int j = 0; j < n_split_buffers; j++) {
            sum = sum + in_buf[i + j * single_buffer_size];
        }
        out_image[i] = sum;
    }
}

/**
 * Applies a luminosity threshold to an image buffer.
 */
std::vector<vec3> apply_threshold(const std::vector<vec3> &in_buf, float threshold) {
    std::vector<vec3> result(in_buf.size());
    for (size_t i = 0; i < in_buf.size(); i++) {
        float l = luminosity(in_buf[i]);
        result[i] = (l < threshold) ? vec3(0.0f) : in_buf[i]; 
    }
    return result;
}

/**
 * Adds two image buffers. 
 */
void image_add(std::vector<vec3> &buf, const std::vector<vec3> &layer, float opacity) {
    assert(buf.size() == layer.size()); 
    for (size_t i = 0; i < buf.size(); i++) {
        buf[i] = buf[i] + opacity * layer[i];
    }
}

