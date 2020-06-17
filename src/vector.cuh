#ifndef VECTOR_H
#define VECTOR_H

#include <math.h>
#include <iostream>

// 2 dimensional int vector
struct ivec2 {
    int x, y;
    __device__ __host__ ivec2() {}
    __device__ __host__ ivec2(int x, int y) {
        this->x = x;
        this->y = y;
    }
};

// 2 dimensional float vector
struct vec2 {
    float x, y;
    __device__ __host__ vec2() {}
    __device__ __host__ vec2(float x, float y) {
        this->x = x;
        this->y = y;
    }
    __device__ __host__ vec2 operator+(const vec2& v) const {
        return vec2(this->x + v.x, this->y + v.y);
    }
    __device__ __host__ vec2 operator-(const vec2& v) const {
        return vec2(this->x - v.x, this->y - v.y);
    }
    __device__ __host__ vec2 normalized() const {
        float norm = sqrtf(this->x * this->x + this->y * this->y);
        return vec2(this->x / norm, this->y / norm);
    }
    __device__ __host__ void normalize() {
        vec2 nv = this->normalized();
        this->x = nv.x;
        this->y = nv.y;
    }
};

// 3 dimensional float vector
struct vec3 {
    float x, y, z;
    __device__ __host__ vec3() {}
    __device__ __host__ vec3(float x, float y, float z) {
        this->x = x;
        this->y = y;
        this->z = z;
    }
    __device__ __host__ vec3 operator+(const vec3& v) const {
        return vec3(this->x + v.x, this->y + v.y, this->z + v.z);
    }
    __device__ __host__ vec3 operator-(const vec3& v) const {
        return vec3(this->x - v.x, this->y - v.y, this->z - v.z);
    }
    __device__ __host__ vec3 operator-() const {
        return vec3(-this->x, -this->y, -this->z);
    }
    __device__ __host__ vec3 normalized() const {
        float norm = sqrtf(this->x * this->x + this->y * this->y + this->z * this->z);
        return vec3(this->x / norm, this->y / norm, this->z / norm);
    }
    __device__ __host__ void normalize() {
        vec3 nv = this->normalized();
        this->x = nv.x;
        this->y = nv.y;
        this->z = nv.z;
    }
};

// vector-vector products
__host__ __device__ inline float dot(const vec2& a, const vec2& b) {
    return a.x * b.x + a.y * b.y;
}
__host__ __device__ inline float dot(const vec3& a, const vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__host__ __device__ inline vec3 cross(const vec3& a, const vec3& b) {
    return vec3(
            (a.y * b.z) - (a.z * b.y),
            (a.z * b.x) - (a.x * b.z),
            (a.x * b.y) - (a.y * b.x)
    );
}

// scalar operations
__device__ __host__ inline vec2 operator*(float s, const vec2& v) {
    return vec2(s * v.x, s * v.y);
}
__device__ __host__ inline vec3 operator*(float s, const vec3& v) {
    return vec3(s * v.x, s * v.y, s * v.z);
}
__device__ __host__ inline vec3 operator*(const vec3& v, float s) {
    return operator*(s, v);
}
__device__ __host__ inline vec3 operator/(const vec3& v, float s) {
    return vec3(v.x / s, v.y / s, v.z / s);
}

// printing
__host__ inline std::ostream& operator<<(std::ostream& os, const ivec2& v) {
    return os << "(" << v.x << ", " << v.y << ")";
}
__host__ inline std::ostream& operator<<(std::ostream& os, const vec2& v) {
    return os << "(" << v.x << ", " << v.y << ")";
}
__host__ inline std::ostream& operator<<(std::ostream& os, const vec3& v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
}

#endif