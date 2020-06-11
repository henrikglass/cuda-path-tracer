#include <iostream>
#include "renderer.cuh"
#include "io.h"

int main(void) {
    Material m(vec3(1,0,0), 0, 0);
    Entity sphere(vec3(0,0,2), 0.5f, m);
    Entity sphere2(vec3(1,1,2), 0.3f, m);
    Scene scene;
    scene.add_entity(sphere);
    scene.add_entity(sphere2);
    ivec2 res = ivec2(1000, 800);
    Camera camera(vec3(0,0,0), vec3(0,0,1), res, res.y);
    Image image = render(camera, scene);
    std::cout << image.resolution << std::endl;
    save_ppm("output.ppm", image);
    //std::cout << render(v6, v7) << std::endl;
    //std::cout << sizeof(vec3) << std::endl;
    //std::cout << sizeof(vec2) << std::endl;
    //std::cout << "Hello World!" << std::endl;
}