#define STB_IMAGE_IMPLEMENTATION
#include "../include/image.h"

Image *read_image(char *path) {
  Image *img = malloc(sizeof(Image));
  if (img == NULL) {
    fprintf(stderr, "Error allocating memory for Image struct\n");
    return NULL;
  }
  img->data = NULL;
  img->name = NULL;
  img->type = NULL;

  char *last_slash = strrchr(path, '/');
  char *file_name = (last_slash != NULL) ? last_slash + 1 : path;

  char *dot = strrchr(file_name, '.');
  if (dot != NULL && dot != file_name) {
    img->type = strdup(dot + 1);
    size_t name_len = (size_t)(dot - file_name);
    img->name = malloc(name_len + 1);
    if (img->name != NULL) {
      memcpy(img->name, file_name, name_len);
      img->name[name_len] = '\0';
    }
  } else {
    img->name = strdup(file_name);
    img->type = NULL;
  }

  unsigned char *image_data =
      stbi_load(path, &img->width, &img->height, &img->channel, 0);
  if (image_data == NULL) {
    fprintf(stderr, "Error loading image: %s\n", stbi_failure_reason());
    free_image(img);
    return NULL;
  }

  int total_pixels = img->width * img->height * img->channel;
  img->data = malloc(total_pixels * sizeof(float));
  if (img->data == NULL) {
    fprintf(stderr, "Error allocating memory for image data\n");
    stbi_image_free(image_data);
    free_image(img);
    return NULL;
  }

  for (int i = 0; i < total_pixels; i++) {
    img->data[i] = image_data[i];
  }

  stbi_image_free(image_data);
  return img;
}

void free_image(Image *img) {
  if (!img) {
    perror("Error freeing image. NULL\n");
    return;
  }
  if (img->name != NULL) {
    free(img->name);
  }
  if (img->type != NULL) {
    free(img->type);
  }
  if (img->data != NULL) {
    free(img->data);
  }

  free(img);
  return;
}

void print_image_info(Image *img) {
  if (!img) {
    perror("Error printing image info. NULL\n");
    return;
  }
  printf("Image name: %s\n", img->name);
  printf("Image type: %s\n", img->type);
  printf("Image width: %d\n", img->width);
  printf("Image height: %d\n", img->height);
  printf("Image channel: %d\n", img->channel);

  printf("Image Shape: %d %d %d\n", img->height, img->width, img->channel);
  return;
}

Matrix *image_to_matrix(Image *img, int target_size) {
  if (img == NULL || img->data == NULL) {
    fprintf(stderr, "Error: NULL image in image_to_matrix\n");
    return NULL;
  }

  Matrix *mat = create_matrix(target_size * target_size, 1);
  if (mat == NULL) {
    return NULL;
  }

  int src_w = img->width;
  int src_h = img->height;
  int ch = img->channel;

  for (int ty = 0; ty < target_size; ty++) {
    for (int tx = 0; tx < target_size; tx++) {
      // Compute source region for this target pixel (area average)
      int src_x0 = tx * src_w / target_size;
      int src_y0 = ty * src_h / target_size;
      int src_x1 = (tx + 1) * src_w / target_size;
      int src_y1 = (ty + 1) * src_h / target_size;

      if (src_x1 == src_x0)
        src_x1 = src_x0 + 1;
      if (src_y1 == src_y0)
        src_y1 = src_y0 + 1;
      if (src_x1 > src_w)
        src_x1 = src_w;
      if (src_y1 > src_h)
        src_y1 = src_h;

      float sum = 0.0f;
      int count = 0;

      for (int sy = src_y0; sy < src_y1; sy++) {
        for (int sx = src_x0; sx < src_x1; sx++) {
          int idx = (sy * src_w + sx) * ch;
          if (ch >= 3) {
            // RGB to grayscale using luminance weights
            sum += 0.299f * img->data[idx] + 0.587f * img->data[idx + 1] +
                   0.114f * img->data[idx + 2];
          } else {
            sum += img->data[idx];
          }
          count++;
        }
      }

      mat->data[ty * target_size + tx] = (sum / count) / 255.0f;
    }
  }

  return mat;
}

int list_image_paths(const char *dir_path, char ***paths_out) {
  DIR *dir = opendir(dir_path);
  if (dir == NULL) {
    fprintf(stderr, "Error: Cannot open directory %s\n", dir_path);
    *paths_out = NULL;
    return 0;
  }

  int count = 0;
  int capacity = 64;
  char **paths = malloc(capacity * sizeof(char *));
  if (paths == NULL) {
    closedir(dir);
    *paths_out = NULL;
    return 0;
  }

  struct dirent *entry;
  while ((entry = readdir(dir)) != NULL) {
    char *dot = strrchr(entry->d_name, '.');
    if (dot == NULL)
      continue;
    if (strcasecmp(dot, ".jpg") != 0 && strcasecmp(dot, ".jpeg") != 0)
      continue;

    size_t path_len = strlen(dir_path) + strlen(entry->d_name) + 2;
    char *full_path = malloc(path_len);
    if (full_path == NULL)
      continue;
    snprintf(full_path, path_len, "%s/%s", dir_path, entry->d_name);

    if (count >= capacity) {
      capacity *= 2;
      char **temp = realloc(paths, capacity * sizeof(char *));
      if (temp == NULL) {
        free(full_path);
        break;
      }
      paths = temp;
    }

    paths[count++] = full_path;
  }

  closedir(dir);
  *paths_out = paths;
  return count;
}
