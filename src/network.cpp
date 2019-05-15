#include <math.h>
#include <assert.h>

#include "common.h"

namespace saga {

Network::Network(int batch_size, bool backprop)
  : batch_size_(batch_size)
  , iter_(0)
  , backprop_(backprop)
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



const Tensor *Network::addLayer(std::shared_ptr<Layer> layer)
{
  workspace_size_ = std::max(workspace_size_, layer->workspaceSize());
  layers_.push_back(layer);
  printf("Added layer: %s\n", layer->name().c_str());
  return layer->output();
}


void Network::forward(const Tensor *input, bool inference)
{
  if(workspace_ == NULL)
    chkCuda(cudaMalloc(&workspace_, workspace_size_));

  for(size_t i = 0; i < layers_.size(); i++) {
    input = layers_[i]->forward(*this, *input, inference);
  }
}

void Network::backprop(const Tensor *input, const Tensor *dy)
{
  for(ssize_t i = layers_.size() - 1; i >= 0; i--) {
    const Tensor *prev = i > 0 ? layers_[i - 1]->output() : input;
    dy = layers_[i]->backprop(*this, *prev, *dy);
  }
  iter_++;
}

std::unique_ptr<Optimizer> Network::makeOptimizer(const Size &s) const {
  return optimizer_factory_(s, *this);
}


}
