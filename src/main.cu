#include <iostream>
#include "renderer.cuh"
#include "io.h"

int main(void) {
    Material m(vec3(1,0,0), 0, 0);
    Entity sphere(vec3(0,0,2), 0.5f, m);
    Entity sphere2(vec3(1,0,2), 0.3f, m);
    Scene scene;
    scene.addEntity(sphere);
    scene.addEntity(sphere2);
    ivec2 res = ivec2(100, 80);
    Camera camera(vec3(0,0,0), vec3(0,0,1), res, res.y);
    Image image = render(camera, scene);
    std::cout << image.resolution << std::endl;
    save_ppm("output.ppm", image);
    //std::cout << render(v6, v7) << std::endl;
    //std::cout << sizeof(vec3) << std::endl;
    //std::cout << sizeof(vec2) << std::endl;
    //std::cout << "Hello World!" << std::endl;
}