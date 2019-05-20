#include <math.h>
#include <assert.h>

#include "common.h"

namespace saga {


static void
logcb(cudnnSeverity_t sev, void *udata, const cudnnDebug_t *dbg,
      const char *msg)
{
  fprintf(stderr, "%s\n", msg);
}

Network::Network(int batch_size, bool backprop)
  : batch_size_(batch_size)
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

  if(0)
    chkCUDNN(cudnnSetCallback(CUDNN_SEV_INFO_EN, this, logcb));

  optimizer_factory_ = &makeGradientDescentOptimizer;
}



const Layer *Network::addLayer(std::shared_ptr<Layer> layer)
{
  workspace_size_ = std::max(workspace_size_, layer->workspaceSize());
  layers_.push_back(layer);
  printf("Added layer: %s %p\n", layer->name().c_str(), layer->gradient());
  return layer.get();
}


void Network::forward(bool inference)
{
  inference_ = inference;

  if(workspace_ == NULL)
    chkCuda(cudaMalloc(&workspace_, workspace_size_));

  for(size_t i = 0; i < layers_.size(); i++) {
    layers_[i]->forward(*this);
  }
}

void Network::backprop(unsigned int iteration)
{
  iteration_ = iteration;
  for(ssize_t i = layers_.size() - 1; i >= 0; i--) {
    layers_[i]->backprop(*this);
  }
}

std::unique_ptr<Optimizer> Network::makeOptimizer(const Size &s) const {
  return optimizer_factory_(s, *this);
}


class Input : public Layer {

public:
  Input(const Tensor *input)
    : output_(input)
  {}

  const Tensor *output() const override {
    return output_;
  }

  std::string name() const override {
    std::stringstream ss;
    ss << "Input " << output_->name();
    return ss.str();
  }

  void forward(const Network &n) {}

private:
  const Tensor *output_;
};

std::shared_ptr<Layer> makeInput(const Tensor *t)
{
  return std::make_shared<Input>(t);
}


}
