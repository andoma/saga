#include <math.h>
#include <assert.h>

#include "common.h"

namespace saga {

Network::Network(int batch_size, bool backprop)
  : batch_size_(batch_size)
  , iter_(0)
  , backprop_(backprop)
  , inference_(false)
  , workspace_(NULL)
  , workspace_size_(0)
{

  int device;
  cudaGetDevice(&device);

  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  printf("  Device name: %s\n", prop.name);
  printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
  printf("  CanMapHostMem: %d\n", prop.canMapHostMemory);
  printf("  ComputeMode: %d\n", prop.computeMode);
  printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
  printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
         2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

  chkCUDNN(cudnnCreate(&cudnn_));
  chkCuda(cublasCreate(&cublas_));

  optimizer_factory_ = &makeGradientDescentOptimizer;
}



std::shared_ptr<Tensor> Network::addLayer(std::shared_ptr<Layer> layer)
{
  workspace_size_ = std::max(workspace_size_, layer->workspaceSize());
  layers_.push_back(layer);
  printf("Added layer: %s\n", layer->name().c_str());
  return layer->output();
}



void Network::forward()
{
  if(workspace_ == NULL)
    chkCuda(cudaMalloc(&workspace_, workspace_size_));

  for(size_t i = 0; i < layers_.size(); i++) {
    layers_[i]->forward(*this);
  }
}

void Network::backprop(std::shared_ptr<Tensor> dy)
{
  for(ssize_t i = layers_.size() - 1; i >= 0; i--) {
    dy = layers_[i]->backprop(*this, dy);
  }
  iter_++;
}

std::unique_ptr<Optimizer> Network::makeOptimizer(const Size &s) const {
  return optimizer_factory_(s, *this);
}


}
