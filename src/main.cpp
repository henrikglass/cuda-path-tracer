#include <iostream>
#include "cuda_runtime.h"
#include "renderer.cuh"
#include "io.h"

int main(void) {
    Material m(vec3(1,0,0), 0, 0, false);
    Entity dragon("examples/dragon/dragon.obj", m);
    dragon.rotate(vec3(0, 3.1415f/4.0f, 0));
    dragon.scale(2.0f);
    dragon.construct_octree();
    Scene scene;

    // test transform
    std::cout << "dragon:" << std::endl;
    dragon.print();

    scene.add_entity(dragon);

    ivec2 res = ivec2(1024, 768);
    //ivec2 res = ivec2(4096, 2160);
    //ivec2 res = ivec2(8, 8);
    Camera camera(vec3(0,0,-2.5f), vec3(0,0,1), res, res.y);
    Image image = render(camera, scene);
    std::cout << image.resolution << std::endl;
    save_ppm("output.ppm", image);

    // test teapot
    /*std::cout << "teapot: " << std::endl;
    teapot.print();

    //teapot.scale(0.02f);
    std::cout << "transforming " << std::endl;
    teapot.translate(vec3(500+205.3 - 0.5f, 500+ 0.5f, -82.3 + 3));
    teapot.rotate(vec3(-3.141592f/1.7f, 3.141592f/8.0f, -3.141592f/8.0f));
    teapot.print();
    std::cout << "done " << std::endl;
    std::cout << "constrcuting octree " << std::endl;
    teapot.construct_octree();
    std::cout << "done " << std::endl;
    //teapot.octree->pretty_print(0);
    //std::cout << "-\n-\n-\n-\n-\n-\n" << std::endl;
    //Vertex v0 = teapot.vertices[teapot.triangles[48].idx_a];
    //Vertex v1 = teapot.vertices[teapot.triangles[48].idx_b];
    //Vertex v2 = teapot.vertices[teapot.triangles[48].idx_c];
    //std::cout << v0.position << " " << v1.position << " " << v2.position << std::endl;
    //std::cout << "nt: " << teapot.n_triangles << std::endl;
    //std::cout << "n0: " << teapot.octree->n_triangle_indices << std::endl;

    //scene.add_entity(sphere);
    //scene.add_entity(sphere2);
    scene.add_entity(teapot);
    //std::cout << sizeof(Octree) << std::endl;


    // rendering
    ivec2 res = ivec2(1024, 768);
    //ivec2 res = ivec2(4096, 2160);
    //ivec2 res = ivec2(8, 8);
    Camera camera(vec3(500,500,-170), vec3(0,0,1), res, res.y);
    Image image = render(camera, scene);
    std::cout << image.resolution << std::endl;
    save_ppm("output.ppm", image);*/


    // test Octree
    //std::cout << sizeof(Octree) << std::endl;
    //Vertex vs[9];
    //// 0 - 0 - 0
    //vs[0] = Vertex(vec3(0.1, 0.1, 0.1), vec3(0, 0, 0));
    //vs[1] = Vertex(vec3(0.1, 0.2, 0.1), vec3(0, 0, 0));
    //vs[2] = Vertex(vec3(0.2, 0.2, 0.1), vec3(0, 0, 0));
    //// 0 - 7
    //vs[3] = Vertex(vec3(0.52, 0.52, 0.60), vec3(0, 0, 0));
    //vs[4] = Vertex(vec3(0.52, 0.82, 0.70), vec3(0, 0, 0));
    //vs[5] = Vertex(vec3(0.82, 0.62, 0.65), vec3(0, 0, 0));
    //// 0 
    //vs[6] = Vertex(vec3(0.52, 0.52, 0.30), vec3(0, 0, 0));
    //vs[7] = Vertex(vec3(0.52, 0.82, 0.70), vec3(0, 0, 0));
    //vs[8] = Vertex(vec3(0.82, 0.62, 0.65), vec3(0, 0, 0));
    //Triangle tr[3];
    //tr[0] = Triangle(0, 1, 2);
    //tr[1] = Triangle(3, 4, 5);
    //tr[2] = Triangle(6, 7, 8);
    //AABB aabb(vec3(0,0,0), vec3(1,1,1)); // min , max
    //Octree oct(aabb, 0);
    //std::cout << oct.triangle_indices.size() << std::endl;
//
    //oct.insert_triangles(vs, tr, 3);
    //oct.pretty_print(0);
    //oct.copy_to_device();
    //std::cout << oct.children[0] << std::endl;






    //std::cout << render(v6, v7) << std::endl;
    //std::cout << sizeof(vec3) << std::endl;
    //std::cout << sizeof(vec2) << std::endl;
    //std::cout << "Hello World!" << std::endl;
}