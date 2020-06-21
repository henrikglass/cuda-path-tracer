#include <iostream>
#include "cuda_runtime.h"
#include "renderer.cuh"
#include "io.h"

int main(void) {
    // materials
    Material m_white(vec3(1,1,1), 0, 0, false);
    Material m_red(vec3(1,0,0), 0, 0, false);
    Material m_green(vec3(0,1,0), 0, 0, false);
    Material m_ball(vec3(1,0,1), 0, 0, false);
    Material m_light(vec3(0,1,1), 20, 0, false);

    // entities
    Entity ball(vec3(27.5f, 27.5f, 0.0f), 10, m_ball);
    Entity ball2(vec3(45.5f, 27.5f, 0.0f), 5, m_ball);
    Entity wall_l("examples/cornellbox/side_wall.obj", m_red);
    Entity wall_r("examples/cornellbox/side_wall.obj", m_green);
    Entity wall_back("examples/cornellbox/back_wall.obj", m_white);
    Entity floor("examples/cornellbox/floor.obj", m_white);
    Entity ceiling("examples/cornellbox/floor.obj", m_white);
    Entity light("examples/cornellbox/floor.obj", m_light);

    // transforms
    wall_r.translate(vec3(55.0f, 0, 0));
    wall_r.rotate(vec3(0, 3.14159265f, 0));
    ceiling.translate(vec3(0, 55.0f, 0));
    ceiling.rotate(vec3(3.14159265f, 0, 0));
    light.translate(vec3(0, 54.5f, 0));
    light.rotate(vec3(3.14159265f, 0, 0));
    wall_l.scale(1.1f);
    wall_r.scale(1.1f);
    wall_back.scale(1.1f);
    floor.scale(1.1f);
    ceiling.scale(1.1f);
    light.scale(0.2f);

    // add to scene scene
    Scene scene;
    scene.add_entity(&wall_l);
    scene.add_entity(&wall_r);
    scene.add_entity(&wall_back);
    scene.add_entity(&floor);
    scene.add_entity(&ceiling);
    scene.add_entity(&ball);
    scene.add_entity(&ball2);
    scene.add_entity(&light);

    // camera
    ivec2 res = ivec2(1024, 768);
    Camera camera(vec3(27.5f, 27.5f,-100), vec3(0,0,1), res, res.y*1.5);

    // render
    Image image = render(camera, scene);
    std::cout << image.resolution << std::endl;
    save_ppm("output.ppm", image);
}