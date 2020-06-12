#ifndef VECTOR_H
#define VECTOR_H

#include <math.h>
#include <iostream>

struct ivec2 {
    int x, y;
    __device__ __host__ ivec2() {}
    __device__ __host__ ivec2(int x, int y) {
        this->x = x;
        this->y = y;
    }
};

struct vec2 {
    float x, y;
    __device__ __host__ vec2() {}
    __device__ __host__ vec2(float x, float y) {
        this->x = x;
        this->y = y;
    }
    __device__ __host__ vec2 operator+(const vec2& v) const;
    __device__ __host__ vec2 operator-(const vec2& v) const;
    __device__ __host__ vec2 normalized() const;
    __device__ __host__ void normalize();
};

struct vec3 {
    float x, y, z;
    __device__ __host__ vec3() {
        // TODO remove
        //this->x = 0;
        //this->y = 0;
        //this->z = 0;
    }
    __device__ __host__ vec3(float x, float y, float z) {
        this->x = x;
        this->y = y;
        this->z = z;
    }
    __device__ __host__ vec3 operator+(const vec3& v) const;
    __device__ __host__ vec3 operator-(const vec3& v) const;
    __device__ __host__ vec3 operator-() const;
    __device__ __host__ vec3 normalized() const;
    __device__ __host__ void normalize();
};

__host__ __device__ float dot(const vec2& a, const vec2& b);
__host__ __device__ float dot(const vec3& a, const vec3& b);

__device__ __host__ vec2 operator*(float s, const vec2& v);
__device__ __host__ vec3 operator*(float s, const vec3& v);
__device__ __host__ vec3 operator*(const vec3& v, float s);

__device__ __host__ vec3 operator/(const vec3& v, float s);

__host__ std::ostream& operator<<(std::ostream& os, const ivec2& v);
__host__ std::ostream& operator<<(std::ostream& os, const vec2& v);
__host__ std::ostream& operator<<(std::ostream& os, const vec3& v);

#endif