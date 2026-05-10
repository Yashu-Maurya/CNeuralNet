#ifndef IMAGE_H
#define IMAGE_H
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"
#include "stb_image.h"

typedef struct {
  float *data;

  char *type; // JPG, PNG etc.
  char *name;

  int width;
  int height;

  int channel;

} Image;

Image *read_image(char *path);
void free_image(Image *img);
void print_image_info(Image *img);
Matrix *image_to_matrix(Image *img, int target_size);
int list_image_paths(const char *dir_path, char ***paths_out);

#endif