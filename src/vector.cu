#include "vector.cuh"

vec2 vec2::operator+(const vec2& v) const {
    return vec2(this->x + v.x, this->y + v.y);
}

vec2 vec2::operator-(const vec2& v) const {
    return vec2(this->x - v.x, this->y - v.y);
}

vec2 vec2::normalized() const {
    float norm = sqrt(this->x * this->x + this->y * this->y);
    return vec2(this->x / norm, this->y / norm);
}

void vec2::normalize() {
    vec2 nv = this->normalized();
    this->x = nv.x;
    this->y = nv.y;
}

vec3 vec3::operator+(const vec3& v) const {
    return vec3(this->x + v.x, this->y + v.y, this->z + v.z);
}

vec3 vec3::operator-(const vec3& v) const {
    return vec3(this->x - v.x, this->y - v.y, this->z - v.z);
}

vec3 vec3::operator-() const {
    return vec3(-this->x, -this->y, -this->z);
}

vec3 vec3::normalized() const {
    float norm = sqrt(this->x * this->x + this->y * this->y + this->z * this->z);
    return vec3(this->x / norm, this->y / norm, this->z / norm);
}

void vec3::normalize() {
    vec3 nv = this->normalized();
    this->x = nv.x;
    this->y = nv.y;
    this->z = nv.z;
}

float dot(const vec2& a, const vec2& b) {
    return a.x * b.x + a.y * b.y;
}
float dot(const vec3& a, const vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

vec2 operator*(float s, const vec2& v) {
    return vec2(s * v.x, s * v.y);
}
vec3 operator*(float s, const vec3& v) {
    return vec3(s * v.x, s * v.y, s * v.z);
}
vec3 operator*(const vec3& v, float s) {
    return operator*(s, v);
}

vec3 operator/(const vec3& v, float s) {
    return vec3(v.x / s, v.y / s, v.z / s);
}

std::ostream& operator<<(std::ostream& os, const ivec2& v) {
    return os << "(" << v.x << ", " << v.y << ")";
}

std::ostream& operator<<(std::ostream& os, const vec2& v) {
    return os << "(" << v.x << ", " << v.y << ")";
}

std::ostream& operator<<(std::ostream& os, const vec3& v) {
    return os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
}