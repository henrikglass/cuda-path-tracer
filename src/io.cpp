#define PRECISION_8 255
#define PRECISION_16 65535

#include "io.h"

/*
 * Modified from: https://github.com/henrikglass/erodr/blob/master/src/io.c
 */
void save_ppm(
        const std::string& filepath,
        const Image& img
) {
    FILE *fp = fopen(filepath.c_str(), "w");

	// write "header
	fputs("P6\n", fp);	
	fputs("# Generated by cuda-path-tracer\n", fp);
	fprintf(fp, "%d %d\n", img.resolution.x, img.resolution.y);	
	fprintf(fp, "%d\n", PRECISION_8);	
	
	// write data.
	for(int i = 0; i < img.resolution.x * img.resolution.y; i++) {
        std::cout << i << ": " << img.buf[i] << std::endl;
		char color[3];
        color[0] = img.buf[i].x * PRECISION_8;
        color[1] = img.buf[i].y * PRECISION_8;
        color[2] = img.buf[i].z * PRECISION_8;
        fwrite(color, sizeof(char), 3, fp);
	}

	fclose(fp);

    return;
}