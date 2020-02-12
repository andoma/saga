// -*-c++-*-

#pragma once

#include <memory>
#include <mutex>

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


namespace saga {

class CudnnContext : public Context,
                     public std::enable_shared_from_this<CudnnContext> {
public:
  CudnnContext()
    : cudnn_(NULL)
    , cublas_(NULL)
  {}

  ~CudnnContext()
  {}

  int init();

  std::shared_ptr<Program> createProgram(const Graph &graph,
                                         const ProgramConfig &pc);

  cudaStream_t stream_;
  cudnnHandle_t cudnn_;
  cublasHandle_t cublas_;

  std::mutex mutex_;
};


}
