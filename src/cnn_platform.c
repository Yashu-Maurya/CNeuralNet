#include "cnn_platform.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef CNN_PLATFORM_ARDUINO
#include "esp_heap_caps.h"
#endif

static CnnLogCallback cnn_log_callback = NULL;

void cnn_set_log_callback(CnnLogCallback callback) {
  cnn_log_callback = callback;
}

void cnn_log(const char *message) {
  if (message == NULL) {
    return;
  }

  if (cnn_log_callback != NULL) {
    cnn_log_callback(message);
    return;
  }

  fputs(message, stdout);
}

void cnn_vlogf(const char *format, va_list args) {
  if (format == NULL) {
    return;
  }

  char buffer[192];
  vsnprintf(buffer, sizeof(buffer), format, args);
  cnn_log(buffer);
}

void cnn_logf(const char *format, ...) {
  va_list args;
  va_start(args, format);
  cnn_vlogf(format, args);
  va_end(args);
}

void *cnn_alloc(size_t size) {
  if (size == 0) {
    return NULL;
  }

#ifdef CNN_PLATFORM_ARDUINO
  void *ptr = heap_caps_malloc(size, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  if (ptr != NULL) {
    return ptr;
  }
  return heap_caps_malloc(size, MALLOC_CAP_8BIT);
#else
  return malloc(size);
#endif
}

void cnn_free(void *ptr) {
  if (ptr == NULL) {
    return;
  }

#ifdef CNN_PLATFORM_ARDUINO
  heap_caps_free(ptr);
#else
  free(ptr);
#endif
}
