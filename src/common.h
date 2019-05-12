// -*-c++-*-

#pragma once

#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>

#include <sstream>

#include "saga.h"

#define chkCUDNN(expression) {                                          \
    const cudnnStatus_t cudnn_status__ = (expression);                  \
    if(cudnn_status__ != CUDNN_STATUS_SUCCESS) {                        \
      fprintf(stderr, "CUDNN error at %s:%d in %s: %s\n",               \
              __FILE__, __LINE__, __FUNCTION__,                         \
              cudnnGetErrorString(cudnn_status__));                     \
      abort();                                                          \
    }                                                                   \
  }


#define chkCuda(expression) {                                           \
    int cuda_status__ = (expression);                                   \
    if(cuda_status__) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d in %s\n",                    \
              __FILE__, __LINE__, __FUNCTION__);                        \
    }                                                                   \
  }


namespace saga {
#if 0
std::string fmt(const char *const fmt, ...)
{
  char *buffer = NULL;
  va_list ap;

  va_start(ap, fmt);
  int r = vasprintf(&buffer, fmt, ap);
  va_end(ap);
  if(r == -1)
    return "";

  std::string result = buffer;
  free(buffer);

  return result;
}
#endif
}
