#ifndef IMAGE_H
#define IMAGE_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
// This struct and it's functions are an abstraction
// of stb_image library. 

typedef struct {
    float *data;

    char *type; // JPG, PNG etc.
    char *name;
} Image ;

Image* read_image(char *path);
void free_image(Image *img);

#endif