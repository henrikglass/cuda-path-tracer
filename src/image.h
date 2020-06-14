#ifndef IMAGE_H
#define IMAGE_H

struct Image {
    std::vector<vec3> buf;
    ivec2 resolution;
    Image(std::vector<vec3> buf, ivec2 resolution) {
        this->buf = buf;
        this->resolution = resolution;
    }
};

#endif