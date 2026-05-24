#include "cnn_platform.h"

#include <stdio.h>

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
