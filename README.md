# cuda-path-tracer
cuda-path-tracer is a naive path tracer implemented using CUDA C++. It was started as a spare time project for me to learn more about graphics programming and GPGPU computation. Please keep in mind this project isn't really designed for anyone to use as an actual tool. There is no front-end, nothing even close to it, and there has been no rigorous testing. You are however free to do anything you want with the code, as long as it's compliant with the MIT license.

![Render of a statue of HC Andersen](/docs/images/rm1.jpg)
*Render of a statue of HC Andersen (1920x1080 resolution, 2048 samples per pixel)*

## Features

### GPU accelerated rendering
Cuda-path-tracer renders scenes by first dividing the image into many small square segments called tiles. Each tile is represented as a cuda thread block and each pixel in a tile is handled by a single cuda thread. Each thread dispatched to the gpu computes 32 samples before completing. To increase the number of samples per pixel, tiles are computed multiple times over with the results being averaged at the end. Cuda-path-tracer also supports rendering to multiple split image buffers simultaneously as a way to further decrease render times in exchange for higher memory usage.

### Materials, Textures & Image based lighting with HDRIs
cuda-path-tracer features a relatively basic material model. Materials can describe color (albedo), specularity, smoothness, light emissiveness and face normal offsets. These properties, with the exception of emissiveness, can be described through textures. Speaking of textures, you can use HDRIs to light your scene!

![Textures](/docs/images/texture_compare.jpg)
*From left to right: No textures applied. Normal map applied. Normal map + albedo map applied*

### GPU accelerated post processing
Cuda-path-tracer supports the application of convolution matrices to images using the GPU. Use cases I have explored include light bloom effects and bilateral filtering to reduce noise for low SPP images.

### Spatial acceleration structures
To render complex objects with large polygon counts within reasonable time cuda-path-tracer gives the option to store entites in octrees. In combination with an efficent ray-octree traversal algorithm, this reduces render times for complex scenes by multiple orders of magnitude.

### Other cool stuff
The camera has adjustable aperture and focus 😎

## Issues
There's some stuff that's not complete or not working correctly. First, there is a bug which causes some triangles to dissappear when using octrees. It's rare, and when it does occur it's usually fixable by adjusting the octree depth. Probably not that difficult of a fix, I just grew tired of this project not to long after I noticed it.  
  
Secondly, face normals are not read correctly when importing an entity as a \*.obj file. This causes issues when setting the smooth shading material property to true. A simple workaround is to let cuda-path-tracer recompute the face normals. The fix is not that difficult, again I just grew tired of the project.

# Building & Running
Just run `make clean`, then `make`. To execute, run `./cudaPathTracer`. I make no guarantees, but it runs on my machine!

# Credits
This project uses open source components, listed below.

Project: tinyobjloader https://github.com/tinyobjloader/tinyobjloader  
Copyright (c) 2012-2019 Syoyo Fujita and many contributors  
License (MIT): https://github.com/tinyobjloader/tinyobjloader/blob/master/LICENSE  
