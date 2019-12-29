#pragma once

#include <cudnn.h>
#include <cublas_v2.h>


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



