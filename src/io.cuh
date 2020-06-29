#ifndef IO_H
#define IO_H

#include <vector>
#include <string>
#include "vector.cuh"
#include "image.h"

void save_ppm(
        const std::string& filepath,
        const Image& img
);

#endif