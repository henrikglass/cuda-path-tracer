#include <iostream>
#include "vector.h"

int main(void) {
    vec2 v1(4.0f, 5.0f);
    vec2 v2(1.1f, 2.0f);
    vec2 v3 = v1 + v2 + v2;
    std::cout << v3.normalized() << std::endl;
    std::cout << "Hello World!" << std::endl;
}