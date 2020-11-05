#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"
#include "renderer.cuh"
#include "io.cuh"

int main(void) {
    // materials
    Material m_white(vec3(1,1,1), vec3(0.0f), 0, 0, false);
    Material m_lg(vec3(0.8f), vec3(0.0f), 0, 0, true);
    Material m_ball(vec3(0.8f,0.8f,0.8f), vec3(0.5f), 0, 0.0f, false);
    Material m_light(vec3(1,1,1), vec3(0.0f), 10, 0, false);
    Material m_default;
    Material m_hc;
    //m_hc.specular = vec3(0.2f);
    //m_hc.smoothness = 1.0f;
    //m_hc.albedo_map.set("examples/hcandersen/textures/diff.jpg");
    //m_hc.smoothness_map.set("examples/hcandersen/textures/gloss.jpg");
    m_hc.normal_map.set("examples/hcandersen/textures/normal.jpg");

    // entities
    Entity floor("examples/cornellbox/floor.obj", &m_default);
    Entity ball(vec3(10, 40, 30), 3.0f, &m_light);
    Entity hcandersen("examples/hcandersen/source/80k.obj", &m_hc);

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
    //scene.add_entity(&ball);
    //scene.add_entity(&floor);

    //scene.set_hdri("examples/hdris/pink_sunrise_4k.hdr");
    scene.use_hdri_smoothing(true);
    scene.set_hdri("examples/hdris/quattro_canti_4k.hdr");
    scene.rotate_hdri(0.85f);

    // camera
    //ivec2 res = ivec2(1024, 768);
    ivec2 res = ivec2(1920, 1080);
    
    //Camera camera(vec3(40.5f, 20.5f, -50), vec3(-0.5, 0, 1).normalized(), res);
    //camera.focal_length = 1.5f;
    //camera.aperture = 1;
    //camera.focus_distance = 85;
    Camera camera(vec3(40.5f, 30.5f, -50), vec3(-0.5, 0, 1).normalized(), res);
    camera.focal_length = 2.5f;
    //camera.focal_length = 1.0f;
    camera.aperture = 1;
    camera.focus_distance = 85;

    // render
    Renderer renderer;
    //renderer.set_samples_per_pixel(1024);
    renderer.set_samples_per_pixel(2048);
    Image image = renderer.render(camera, scene);
    std::cout << image.resolution << std::endl;
    save_ppm("output.ppm", image);
}
