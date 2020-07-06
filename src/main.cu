#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"
#include "renderer.cuh"
#include "io.cuh"

int main(void) {
    // materials
    Material m_white(vec3(1,1,1), vec3(0.0f), 0, 0, false);
    Material m_lg(vec3(0.8f), vec3(0.0f), 0, 0, false);
    Material m_ball(vec3(0.3f,0.3f,0.3f), vec3(0.7f, 0.7f, 0.7f), 0, 0.8f, false);

    // entities
    Entity floor("examples/cornellbox/floor.obj", m_lg);
    Entity hcandersen("examples/hcandersen/source/80k.obj", m_lg);

    // transforms
    floor.scale(100.1f);
    hcandersen.rotate(vec3(-3.14159265f / 2.0f, 3.14159265f / 2.0f, 0));
    hcandersen.translate(vec3(0, 22.65, 0));

    // construct octree
    hcandersen.construct_octree();

    // debug
    hcandersen.print();

    // add to scene scene
    Scene scene;
    scene.add_entity(&hcandersen);
    scene.add_entity(&floor);

    scene.set_hdri("examples/hdris/pink_sunrise_4k.hdr");
    scene.rotate_hdri(0.85f);
    //scene.set_hdri_exposure(0.3f);
    //scene.set_hdri_contrast(1.2f);
    //scene.use_hdri_smoothing(true);

    // camera
    ivec2 res = ivec2(1024, 768);
    //ivec2 res = ivec2(2560, 1440);
    //ivec2 res = ivec2(3860, 2160);
    //ivec2 res = ivec2(320, 180);
    //Camera camera(vec3(30, 26.5695f, -100), vec3(0, 0, 1).normalized(), res, 1.2f);
    Camera camera(vec3(40.5f, 27.5f, -50), vec3(-0.5, 0, 1).normalized(), res);
    camera.focal_length = 1.5f;
    //camera.aperture = 6;
    //camera.focus_distance = 140;
    //Camera camera(vec3(-75.5f, 27.5f, -100), vec3(0.5, 0, 1).normalized(), res, 1.2f);

    // render
    Image image = render(camera, scene);
    std::cout << image.resolution << std::endl;
    save_ppm("output.ppm", image);
}