#include "../include/image.h"

Image* read_image(char *path) {
    Image *img = malloc(sizeof(Image));
    char *last_slash = strrchr(path, '/');
    char *file_name;

    if (last_slash != NULL) {
        file_name = last_slash + 1;
    } else {
        file_name = path;
    }
    // TODO: implement image read functions from stb_image.h
    return img;
}

void free_image(Image *img) {
    if(!img) {
        perror("Error freeing image. NULL\n");
        return;
    }
    if(img->name != NULL){
        free(img->name);
    }
    if(img->type != NULL) {
        free(img->type);
    }
    if(img->data != NULL) {
        free(img->data);
    }

    free(img);
    return;
}