#include <iostream>
#include "vector.cuh"
#include "renderer.cuh"

int main(void) {
    vec3 v6(1.0f, 0.0f, 0.0f);
    vec3 v7(1.0f, 1.0f, 0.0f);
    v7.normalize();
    vec3 ans = render(v6, v7);
    std::cout << ans << std::endl;
    //std::cout << render(v6, v7) << std::endl;
    //std::cout << sizeof(vec3) << std::endl;
    //std::cout << sizeof(vec2) << std::endl;
    //std::cout << "Hello World!" << std::endl;
}