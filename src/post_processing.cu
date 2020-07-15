#include "post_processing.cuh"
#include <assert.h>
#include <cmath>
#include <vector>

/*
 * Kernel
 */

void Kernel::print() {
    float sum = 0.0f;
    for (size_t y = 0; y < size*2 + 1; y++) {
        for (size_t x = 0; x < size*2 + 1; x++) {
            printf("%d ", int(10 * buf[y * (2*size + 1) + x])); 
            sum += buf[y * (2*size + 1) + x];
        }
        printf("\n");
    }
    printf("sum: %g\n", sum);
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
    assert(this->buf == nullptr);
    this->size = _size;
    size_t buf_size = (2 * size + 1) * (2 * size + 1);
    this->buf = new float[buf_size];
    int ssize = _size;
    for (int y = 0; y < ssize*2 + 1; y++) {
        for (int x = 0; x < ssize*2 + 1; x++) {
            size_t idx = y * (this->size * 2  + 1) + x;
            float sample_pos_x = float(x - ssize) / float(ssize);
            float sample_pos_y = float(y - ssize) / float(ssize);
            buf[idx] = sample_gaussian(sample_pos_x, sample_pos_y, sigma);
        }
    } 
}

void Kernel::make_mean(unsigned int _size) {
    assert(this->buf == nullptr);
    this->size = _size;
    size_t buf_size = (2 * size + 1) * (2 * size + 1);
    this->buf = new float[buf_size];
    for (size_t i = 0; i < buf_size; i++) {
        buf[i] = 1.0f;
    }
}

float Kernel::at(int x, int y) const {
    y = this->size + y;
    x = this->size + x;
    size_t idx = y * (this->size * 2 + 1) + x;
    return this->buf[idx];
}


void apply_filter(std::vector<vec3> &buf, ivec2 resolution, const Kernel &kernel) {
    std::vector<vec3> out_buf(buf.size());
    int width  = resolution.x; 
    int height = resolution.y; 
    int ks = kernel.size;
    for (int i = 0 ; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float kernel_weight = 0; // @Incomplete TODO precompute kernel weight and subtract pixels outside image bounds 
            vec3 result(0.0f);
            for (int y = -ks; y <= ks; y++) {
                for (int x = -ks; x <= ks; x++) {
                    int img_y = i + y;
                    int img_x = j + x;
                    size_t img_idx = img_y * width + img_x;
                    if (img_y < 0 || img_y >= height || img_x < 0 || img_x >= width)
                        continue; // out of image bounds 
                    result = result + kernel.at(x, y) * buf[img_idx]; 
                    kernel_weight += kernel.at(x, y);
                }
            }
            result = result / kernel_weight; 
            out_buf[i * width + j] = result;
        }
    }
    buf = out_buf;
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

void image_add(std::vector<vec3> &buf, const std::vector<vec3> &layer) {
    assert(buf.size() == layer.size()); 
    for (size_t i = 0; i < buf.size(); i++) {
        buf[i] = buf[i] + layer[i];
    }
}

