#include <iostream>
#include "cuda_runtime.h"
#include "cuda.h"
#include "renderer.cuh"
#include "io.cuh"

int main(void) {
    // materials
    Material m_white(vec3(1,1,1), vec3(0.0f), 0, 0, false);
    Material m_red(vec3(1,0,0), vec3(0.0f), 0, 0, false);
    Material m_green(vec3(0,1,0), vec3(0.0f), 0, 0, false);
    Material m_ball(vec3(0.3f,0.3f,0.3f), vec3(0.7f, 0.7f, 0.7f), 0, 0.8f, false);
    Material m_light(vec3(1,1,1), vec3(0.0f), 10, 0, false);
    Material m_light0(vec3(1,0,0), vec3(0.0f), 10, 0, false);
    Material m_light1(vec3(0,1,0), vec3(0.0f), 10, 0, false);
    Material m_light2(vec3(0,0,1), vec3(0.0f), 10, 0, false);

    // entities
    Entity ball(vec3(13.5f, 10.01f, 37.5f), 10, m_ball); //m_ball
    Entity ball2(vec3(45.5f, 5.01f, 27.5f), 5, m_white); //m_ball
    Entity wall_l("examples/cornellbox/side_wall.obj", m_green);
    Entity wall_r("examples/cornellbox/side_wall.obj", m_red);
    Entity wall_back("examples/cornellbox/back_wall.obj", m_white); // m_white
    Entity floor("examples/cornellbox/floor.obj", m_white);
    Entity ceiling("examples/cornellbox/floor.obj", m_white);
    Entity light("examples/cornellbox/floor.obj", m_light);
    Entity dragon("examples/dragon/dragon.obj", m_white);
    //Entity light0("examples/cornellbox/floor.obj", m_light0);
    //Entity light1("examples/cornellbox/floor.obj", m_light1);
    //Entity light2("examples/cornellbox/floor.obj", m_light2);

    // transforms
    wall_r.translate(vec3(55.0f, 0, 0));
    wall_r.rotate(vec3(0, 3.14159265f, 0));
    ceiling.translate(vec3(0, 55.0f, 0));
    ceiling.rotate(vec3(3.14159265f, 0, 0));
    light.translate(vec3(0, 54.5f, 0));
    light.rotate(vec3(3.14159265f, 0, 0));
    //light0.translate(vec3(-18, 54.5f, 0));
    //light0.rotate(vec3(3.14159265f, 0, 0));
    //light1.translate(vec3(0, 54.5f, 0));
    //light1.rotate(vec3(3.14159265f, 0, 0));
    //light2.translate(vec3(18, 54.5f, 0));
    //light2.rotate(vec3(3.14159265f, 0, 0));
    wall_l.scale(1.1f);
    wall_r.scale(1.1f);
    wall_back.scale(1.1f);
    //floor.scale(1.1f);
    floor.scale(30.1f);
    ceiling.scale(1.1f);
    light.scale(0.35f);
    dragon.scale(75.0f);
    dragon.rotate(vec3(0, -3.14159265f / 1.8f, 0));
    dragon.translate(vec3(30, 26.5, 16));
    //dragon.translate(vec3(40, 26.5, 16));
    //light0.scale(0.25f);
    //light1.scale(0.25f);
    //light2.scale(0.25f);

    // add to scene scene
    Scene scene;
    //scene.add_entity(&wall_l);
    //scene.add_entity(&wall_r);
    //scene.add_entity(&wall_back);
    dragon.construct_octree();
    floor.print();
    dragon.print();
    //dragon.octree->pretty_print(0);
    //exit(0);
    scene.add_entity(&dragon);
    scene.add_entity(&floor);
    //scene.add_entity(&ceiling);
    //scene.add_entity(&ball);
    //scene.add_entity(&ball2);
    //scene.add_entity(&light);
    //scene.set_hdri("examples/hdris/photostudio.hdr");
    //scene.set_hdri("examples/hdris/piazza_bologni_4k.hdr");
    //scene.set_hdri("examples/hdris/old_bus_depot_1k.hdr");
    //scene.set_hdri("examples/hdris/old_bus_depot_16k.hdr");
    scene.set_hdri("examples/hdris/pink_sunrise_4k.hdr");
    //scene.set_hdri_exposure(0.3f);
    scene.set_hdri_contrast(1.2f);
    //scene.use_hdri_smoothing(true);
    //std::cout << scene.sample_hdri(vec3(0, 0, 1.0f).normalized()) << std::endl;
    //std::cout << scene.sample_hdri(vec3(0.1f, 1.0f, 0.1f).normalized()) << std::endl;
    //std::cout << scene.sample_hdri(vec3(-0.1f, -0.1f, 1.0f).normalized()) << std::endl;
    //std::cout << scene.sample_hdri(vec3(1.0f, 0, 0).normalized()) << std::endl;
    //scene.add_entity(&light0);
    //scene.add_entity(&light1);
    //scene.add_entity(&light2);

    // camera
    ivec2 res = ivec2(1024, 768);
    //ivec2 res = ivec2(320, 180);
    //Camera camera(vec3(30, 26.5695f, -100), vec3(0, 0, 1).normalized(), res, 1.2f);
    Camera camera(vec3(127.5f, 27.5f, -100), vec3(-0.5, 0, 1).normalized(), res, 1.2f);
    //Camera camera(vec3(-75.5f, 27.5f, -100), vec3(0.5, 0, 1).normalized(), res, 1.2f);

    // render
    Image image = render(camera, scene);
    std::cout << image.resolution << std::endl;
    save_ppm("output.ppm", image);
}