#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../../include/image.h"
#include "../../include/network.h"

#define IMG_SIZE 32
#define INPUT_SIZE (IMG_SIZE * IMG_SIZE) // 1024
#define HIDDEN_SIZE_1 512
#define HIDDEN_SIZE_2 256
// #define HIDDEN_SIZE_3 128

#define OUTPUT_SIZE 1

#define LEARNING_RATE 0.01f
#define BATCH_SIZE 1000
#define EPOCHS_PER_BATCH 5

#define DATA_BASE                                                              \
  "../examples/classification_example/Hotdog Not Hotdog "                      \
  "Archive/hotdog-nothotdog/hotdog-nothotdog"

// Shuffle two parallel arrays using Fisher-Yates
static void shuffle_paths(char **paths, int *labels, int n) {
  for (int i = n - 1; i > 0; i--) {
    int j = rand() % (i + 1);
    char *tmp = paths[i];
    paths[i] = paths[j];
    paths[j] = tmp;
    int tl = labels[i];
    labels[i] = labels[j];
    labels[j] = tl;
  }
}

// Train one batch: load images, train for EPOCHS_PER_BATCH, free images
static void train_batch(Network *network, char **paths, int *labels, int count,
                        int batch_num) {
  Matrix *target = create_matrix(OUTPUT_SIZE, 1);

  for (int epoch = 0; epoch < EPOCHS_PER_BATCH; epoch++) {
    int correct = 0;
    float total_loss = 0.0f;

    for (int s = 0; s < count; s++) {
      Image *img = read_image(paths[s]);
      if (img == NULL)
        continue;

      Matrix *input = image_to_matrix(img, IMG_SIZE);
      free_image(img);
      if (input == NULL)
        continue;

      target->data[0] = (float)labels[s];

      Matrix *pred = predict_network(network, input);
      int predicted = (pred->data[0] >= 0.5f) ? 1 : 0;
      if (predicted == labels[s])
        correct++;
      float diff = pred->data[0] - target->data[0];
      total_loss += diff * diff;
      free_matrix(pred);

      train_network(network, input, target, LEARNING_RATE);
      free_matrix(input);
    }

    float accuracy = (float)correct / count * 100.0f;
    float avg_loss = total_loss / count;
    printf("  Batch %d | Epoch %d/%d - Accuracy: %.2f%% - Loss: %.4f\n",
           batch_num, epoch + 1, EPOCHS_PER_BATCH, accuracy, avg_loss);
  }

  free_matrix(target);
}

int main() {
  srand((unsigned int)time(NULL));

  printf("=== Hotdog / Not-Hotdog Classification ===\n\n");
  //   printf("Network: %d -> %d (ReLU) -> %d (Sigmoid)\n", INPUT_SIZE,
  //   HIDDEN_SIZE,
  //          OUTPUT_SIZE);
  printf("Image resize: %dx%d grayscale\n", IMG_SIZE, IMG_SIZE);
  printf("Learning Rate: %.4f\n", LEARNING_RATE);
  printf("Batch size: %d images | Epochs per batch: %d\n\n", BATCH_SIZE,
         EPOCHS_PER_BATCH);

  // --- Collect all training file paths ---
  printf("Scanning directories...\n");

  char **hd_paths = NULL;
  int n_hd = list_image_paths(DATA_BASE "/train/hotdog", &hd_paths);
  printf("  Found %d hotdog training images\n", n_hd);

  char **nhd_paths = NULL;
  int n_nhd = list_image_paths(DATA_BASE "/train/nothotdog", &nhd_paths);
  printf("  Found %d not-hotdog training images\n", n_nhd);

  if (n_hd == 0 || n_nhd == 0) {
    fprintf(stderr, "Error: No images found. Run from the build directory.\n");
    return -1;
  }

  // Merge all training paths + labels into one array
  int total_train = n_hd + n_nhd;
  char **train_paths = malloc(total_train * sizeof(char *));
  int *train_labels = malloc(total_train * sizeof(int));

  for (int i = 0; i < n_hd; i++) {
    train_paths[i] = hd_paths[i];
    train_labels[i] = 1;
  }
  for (int i = 0; i < n_nhd; i++) {
    train_paths[n_hd + i] = nhd_paths[i];
    train_labels[n_hd + i] = 0;
  }

  // Shuffle everything once before batching
  shuffle_paths(train_paths, train_labels, total_train);

  printf("  Total training images: %d\n\n", total_train);

  // --- Create Network ---
  Network *network = create_network();
  add_layer(network, layer_create_dense(INPUT_SIZE, HIDDEN_SIZE_1));
  add_layer(network, layer_create_relu());
  add_layer(network, layer_create_dense(HIDDEN_SIZE_1, HIDDEN_SIZE_2));
  add_layer(network, layer_create_relu());
  add_layer(network, layer_create_dense(HIDDEN_SIZE_2, OUTPUT_SIZE));
  add_layer(network, layer_create_sigmoid());

  print_network_info(network);
  printf("\n");

  // --- Training in batches of BATCH_SIZE ---
  printf("--- Training Phase ---\n");
  int num_batches = (total_train + BATCH_SIZE - 1) / BATCH_SIZE;
  printf("Training %d images in %d batches of %d, %d epochs each\n\n",
         total_train, num_batches, BATCH_SIZE, EPOCHS_PER_BATCH);

  for (int b = 0; b < num_batches; b++) {
    int start = b * BATCH_SIZE;
    int end = start + BATCH_SIZE;
    if (end > total_train)
      end = total_train;
    int count = end - start;

    printf("Batch %d/%d (%d images, samples %d-%d)\n", b + 1, num_batches,
           count, start + 1, end);

    train_batch(network, &train_paths[start], &train_labels[start], count,
                b + 1);
    printf("\n");
  }

  // --- Testing in batches of BATCH_SIZE ---
  printf("--- Testing Phase ---\n");

  char **test_hd_paths = NULL;
  int n_test_hd = list_image_paths(DATA_BASE "/test/hotdog", &test_hd_paths);
  char **test_nhd_paths = NULL;
  int n_test_nhd =
      list_image_paths(DATA_BASE "/test/nothotdog", &test_nhd_paths);

  int total_test = n_test_hd + n_test_nhd;
  char **test_paths = malloc(total_test * sizeof(char *));
  int *test_labels = malloc(total_test * sizeof(int));

  for (int i = 0; i < n_test_hd; i++) {
    test_paths[i] = test_hd_paths[i];
    test_labels[i] = 1;
  }
  for (int i = 0; i < n_test_nhd; i++) {
    test_paths[n_test_hd + i] = test_nhd_paths[i];
    test_labels[n_test_hd + i] = 0;
  }

  printf("Testing on %d images (%d hotdog, %d not-hotdog)\n\n", total_test,
         n_test_hd, n_test_nhd);

  int test_correct = 0;
  int tp = 0, fp = 0, tn = 0, fn = 0;
  int test_batches = (total_test + BATCH_SIZE - 1) / BATCH_SIZE;

  for (int b = 0; b < test_batches; b++) {
    int start = b * BATCH_SIZE;
    int end = start + BATCH_SIZE;
    if (end > total_test)
      end = total_test;

    printf("  Test batch %d/%d (samples %d-%d)...\n", b + 1, test_batches,
           start + 1, end);

    for (int s = start; s < end; s++) {
      Image *img = read_image(test_paths[s]);
      if (img == NULL)
        continue;

      Matrix *input = image_to_matrix(img, IMG_SIZE);
      free_image(img);
      if (input == NULL)
        continue;

      Matrix *pred = predict_network(network, input);
      int predicted = (pred->data[0] >= 0.5f) ? 1 : 0;
      int actual = test_labels[s];

      if (predicted == actual)
        test_correct++;
      if (predicted == 1 && actual == 1)
        tp++;
      if (predicted == 1 && actual == 0)
        fp++;
      if (predicted == 0 && actual == 0)
        tn++;
      if (predicted == 0 && actual == 1)
        fn++;

      free_matrix(pred);
      free_matrix(input);
    }
  }

  float test_accuracy = (float)test_correct / total_test * 100.0f;

  printf("\n=== Results ===\n");
  printf("Test Accuracy: %d/%d = %.2f%%\n", test_correct, total_test,
         test_accuracy);
  printf("\nConfusion Matrix:\n");
  printf("                Predicted HD  Predicted Not-HD\n");
  printf("  Actual HD:       %4d           %4d\n", tp, fn);
  printf("  Actual Not-HD:   %4d           %4d\n", fp, tn);

  if (tp + fp > 0)
    printf("\nPrecision: %.2f%%\n", (float)tp / (tp + fp) * 100.0f);
  if (tp + fn > 0)
    printf("Recall:    %.2f%%\n", (float)tp / (tp + fn) * 100.0f);

  // --- Cleanup ---
  free_network(network);
  for (int i = 0; i < n_hd; i++)
    free(hd_paths[i]);
  for (int i = 0; i < n_nhd; i++)
    free(nhd_paths[i]);
  for (int i = 0; i < n_test_hd; i++)
    free(test_hd_paths[i]);
  for (int i = 0; i < n_test_nhd; i++)
    free(test_nhd_paths[i]);
  free(hd_paths);
  free(nhd_paths);
  free(test_hd_paths);
  free(test_nhd_paths);
  free(train_paths);
  free(train_labels);
  free(test_paths);
  free(test_labels);

  printf("\nDone!\n");
  return 0;
}