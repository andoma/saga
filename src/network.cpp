#include <math.h>
#include <assert.h>

#include "common.h"

namespace saga {

Network::Network(const Tensor &input, bool backprop)
  : batch_size_(input.size().n)
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

}



std::shared_ptr<Tensor> Network::addLayer(std::shared_ptr<Layer> layer)
{
  layers_.push_back(layer);
  return layer->output();
}



void Network::initialize()
{
  for(const auto &l : layers_) {
    printf("Layer %s workspace_size:%zd\n",
           l->name().c_str(), l->workspaceSize());
  }
}


}


