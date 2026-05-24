#ifndef CNN_PLATFORM_H
#define CNN_PLATFORM_H

#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*CnnLogCallback)(const char *message);

void cnn_set_log_callback(CnnLogCallback callback);
void cnn_log(const char *message);
void cnn_logf(const char *format, ...);
void cnn_vlogf(const char *format, va_list args);

#ifdef __cplusplus
}
#endif

#endif
