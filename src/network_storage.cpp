#ifdef ARDUINO

extern "C" {
#include "network.h"
}
#include "cnn_platform.h"

#include <Arduino.h>
#include <FS.h>

static bool write_exact(File &file, const void *data, size_t length) {
  return file.write((const uint8_t *)data, length) == length;
}

static bool read_exact(File &file, void *data, size_t length) {
  return file.read((uint8_t *)data, length) == length;
}

static bool save_matrix_file(Matrix *matrix, File &file) {
  if (matrix == NULL || matrix->data == NULL) {
    return false;
  }

  if (!write_exact(file, &matrix->rows, sizeof(int))) {
    return false;
  }
  if (!write_exact(file, &matrix->columns, sizeof(int))) {
    return false;
  }

  size_t data_length = sizeof(float) * matrix->rows * matrix->columns;
  return write_exact(file, matrix->data, data_length);
}

static bool load_matrix_file(Matrix *matrix, File &file) {
  if (matrix == NULL || matrix->data == NULL) {
    return false;
  }

  int rows = 0;
  int columns = 0;
  if (!read_exact(file, &rows, sizeof(int))) {
    return false;
  }
  if (!read_exact(file, &columns, sizeof(int))) {
    return false;
  }

  if (rows != matrix->rows || columns != matrix->columns) {
    cnn_logf("Stored matrix shape %dx%d did not match expected %dx%d\n", rows,
             columns, matrix->rows, matrix->columns);
    return false;
  }

  size_t data_length = sizeof(float) * matrix->rows * matrix->columns;
  return read_exact(file, matrix->data, data_length);
}

#define LAYER_DENSE 0
#define LAYER_SIGMOID 1
#define LAYER_RELU 2
#define LAYER_CONV2D 3

static int get_layer_type(Layer *layer) {
  if (layer == NULL || layer->name == NULL) {
    return -1;
  }
  if (strcmp(layer->name, "Dense") == 0) return LAYER_DENSE;
  if (strcmp(layer->name, "Sigmoid") == 0) return LAYER_SIGMOID;
  if (strcmp(layer->name, "ReLU") == 0) return LAYER_RELU;
  if (strcmp(layer->name, "Conv2d") == 0) return LAYER_CONV2D;
  return -1;
}

static bool save_layer_file(Layer *layer, File &file) {
  if (layer == NULL) {
    return false;
  }

  int type = get_layer_type(layer);
  if (!write_exact(file, &type, sizeof(int))) {
    return false;
  }

  if (type == LAYER_DENSE) {
    if (!write_exact(file, &layer->input_n, sizeof(int))) {
      return false;
    }
    if (!write_exact(file, &layer->output_n, sizeof(int))) {
      return false;
    }
    if (!save_matrix_file(layer->weights, file)) {
      return false;
    }
    if (!save_matrix_file(layer->bias, file)) {
      return false;
    }
  } else if (type == LAYER_CONV2D) {
    if (!write_exact(file, &layer->input_rows, sizeof(int))) {
      return false;
    }
    if (!write_exact(file, &layer->input_columns, sizeof(int))) {
      return false;
    }
    if (!write_exact(file, &layer->output_rows, sizeof(int))) {
      return false;
    }
    if (!write_exact(file, &layer->output_columns, sizeof(int))) {
      return false;
    }
    if (!save_matrix_file(layer->weights, file)) {
      return false;
    }
    if (!save_matrix_file(layer->bias, file)) {
      return false;
    }
  }

  return true;
}

static Layer *load_layer_file(File &file) {
  int type = -1;
  if (!read_exact(file, &type, sizeof(int))) {
    return NULL;
  }

  if (type == LAYER_DENSE) {
    int input_n = 0;
    int output_n = 0;
    if (!read_exact(file, &input_n, sizeof(int))) {
      return NULL;
    }
    if (!read_exact(file, &output_n, sizeof(int))) {
      return NULL;
    }

    Layer *layer = layer_create_dense(input_n, output_n);
    if (layer == NULL) {
      return NULL;
    }
    if (!load_matrix_file(layer->weights, file) ||
        !load_matrix_file(layer->bias, file)) {
      free_layer(layer);
      return NULL;
    }
    return layer;
  }

  if (type == LAYER_SIGMOID) {
    return layer_create_sigmoid();
  }

  if (type == LAYER_RELU) {
    return layer_create_relu();
  }

  if (type == LAYER_CONV2D) {
    int input_rows = 0;
    int input_columns = 0;
    int output_rows = 0;
    int output_columns = 0;
    if (!read_exact(file, &input_rows, sizeof(int)) ||
        !read_exact(file, &input_columns, sizeof(int)) ||
        !read_exact(file, &output_rows, sizeof(int)) ||
        !read_exact(file, &output_columns, sizeof(int))) {
      return NULL;
    }

    Layer *layer = layer_create_conv2d_shape(
        input_rows, input_columns, input_rows - output_rows + 1,
        input_columns - output_columns + 1);
    if (layer == NULL) {
      return NULL;
    }
    if (!load_matrix_file(layer->weights, file) ||
        !load_matrix_file(layer->bias, file)) {
      free_layer(layer);
      return NULL;
    }
    return layer;
  }

  cnn_logf("Unknown layer type in model file: %d\n", type);
  return NULL;
}

int save_network_fs(Network *network, fs::FS &filesystem, const char *path) {
  if (network == NULL || path == NULL || network->layer_count == 0) {
    return -1;
  }

  File file = filesystem.open(path, FILE_WRITE);
  if (!file) {
    cnn_logf("Could not open %s for writing\n", path);
    return -1;
  }

  bool ok = write_exact(file, &network->layer_count, sizeof(int));
  for (int i = 0; ok && i < network->layer_count; i++) {
    ok = save_layer_file(network->layers[i], file);
  }

  file.close();
  return ok ? 0 : -1;
}

void load_network_fs(Network *network, fs::FS &filesystem, const char *path) {
  if (network == NULL || path == NULL) {
    return;
  }

  File file = filesystem.open(path, FILE_READ);
  if (!file) {
    cnn_logf("Could not open %s for reading\n", path);
    return;
  }

  int layer_count = 0;
  if (!read_exact(file, &layer_count, sizeof(int))) {
    cnn_log("Could not read model header\n");
    file.close();
    return;
  }

  for (int i = 0; i < layer_count; i++) {
    Layer *layer = load_layer_file(file);
    if (layer == NULL) {
      cnn_logf("Could not load layer %d\n", i);
      break;
    }
    add_layer(network, layer);
  }

  file.close();
}

#endif
