#include "vector.h"

vec2 vec2::operator+(const vec2& v) const {
    return vec2(this->x + v.x, this->y + v.y);
}
vec2 vec2::normalized() const {
    float norm = sqrt(this->x * this->x + this->y * this->y);
    return vec2(this->x / norm, this->y / norm);
}

std::ostream& operator<<(std::ostream& os, const vec2& v) {
    return os << "(" << v.x << ", " << v.y << ")";
}