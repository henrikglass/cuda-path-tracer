#ifndef MATH_UTIL_H
#define MATH_UTIL_H

#define C_PI 3.1415926535f
#define C_E 2.7182818284f


template<typename T>
__host__ __device__ T bary_lerp(T a, T b, T c, float u, float v) {
    float w = 1.0f - (u + v);
    return u * b + v * c + w * a;
}

template<typename T>
__host__ __device__ T lerp(T a, T b, float s) {
    return (1.0f - s) * a + s * b;
}

template<typename T>
__host__ __device__ T bilerp(T ul, T ur, T ll, T lr, float u, float v) {
    T left = lerp(ul, ll, v);
    T right = lerp(ur, lr, v);
    return lerp(left, right, u);
}

#endif
