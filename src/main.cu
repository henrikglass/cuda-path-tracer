#include <iostream>
#include "renderer.cuh"
#include "io.h"

int main(void) {
    Material m(vec3(1,0,0), 0, 0, true);
    Entity sphere(vec3(0,0,2), 0.5f, m);
    Entity sphere2(vec3(1,1,2), 0.3f, m);
    Entity teapot("examples/teapot/utah.obj", m);
    Scene scene;

    // test transform
    sphere.scale(0.5f);
    sphere.translate(vec3(0.5, 0, 0));

    // test teapot
    std::cout << "teapot: " << std::endl;
    teapot.print();

    //teapot.scale(0.02f);
    teapot.translate(vec3(500+205.3 - 0.5f, 500+ 0.5f, -82.3 + 3));
    teapot.rotate(vec3(-3.141592f/2.0f, 0, 0));
    teapot.print();

    //scene.add_entity(sphere);
    //scene.add_entity(sphere2);
    scene.add_entity(teapot);


    ivec2 res = ivec2(1024, 768);
    ////ivec2 res = ivec2(8, 8);
    Camera camera(vec3(500,500,-170), vec3(0,0,1), res, res.y);
    Image image = render(camera, scene);
    std::cout << image.resolution << std::endl;
    save_ppm("output.ppm", image);


    //std::cout << render(v6, v7) << std::endl;
    //std::cout << sizeof(vec3) << std::endl;
    //std::cout << sizeof(vec2) << std::endl;
    //std::cout << "Hello World!" << std::endl;
}