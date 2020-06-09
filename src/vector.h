#ifndef VECTOR_H
#define VECTOR_H
#include <math.h>
#include <iostream>

struct vec2 {
    float x, y;
    vec2(float x, float y) {
        this->x = x;
        this->y = y;
    }
    vec2 operator+(const vec2& v) const;
    vec2 normalized() const;
};

std::ostream& operator<<(std::ostream& os, const vec2& v);
#endif