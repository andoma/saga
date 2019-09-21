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

  const float learning_rate = 3e-4;
  setOptimizer(std::bind(makeAdamOptimizer,
                         std::placeholders::_1,
                         std::placeholders::_2,
                         learning_rate));
}



std::shared_ptr<Layer> Network::addLayer(std::shared_ptr<Layer> layer)
{
  layers_.push_back(layer);
  printf("Added layer: %s\n", layer->name().c_str());
  return layer;
}


std::shared_ptr<Layer> Network::nameLayer(std::shared_ptr<Layer> layer,
                                          const std::string &name)
{
  named_layers_[name] = layer;
  return layer;
}


std::shared_ptr<Layer> Network::findLayer(const std::string &name) const
{
  auto r = named_layers_.find(name);
  if(r == named_layers_.end())
    return nullptr;
  return r->second;
}

void Network::forward(bool inference)
{
  inference_ = inference;

  if(workspace_ == NULL) {
    auto last = layers_[layers_.size() - 1];

    last->output()->allocate();
    if(backprop_)
      last->gradient()->allocate();

    for(size_t i = 0; i < layers_.size(); i++) {
      printf("Setup layer: %s\n", layers_[i]->name().c_str());
      layers_[i]->setup(*this);
      workspace_size_ = std::max(workspace_size_, layers_[i]->workspaceSize());
    }
    chkCuda(cudaMalloc(&workspace_, workspace_size_));
    printf("workspace: %zd\n", workspace_size_);
  }

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
  Input(Tensor *input, bool with_grad)
    : output_(input)
    , output_grad_(with_grad ? std::make_unique<Tensor>(*input) : nullptr)
  {}

  Tensor *output() const override {
    return output_;
  }

  Tensor *gradient() const override {
    return output_grad_.get();
  }

  std::string name() const override {
    std::stringstream ss;
    ss << "Input " << output_->name();
    return ss.str();
  }

  void forward(const Network &n) {
  }

private:
  Tensor *output_;
  std::unique_ptr<Tensor> output_grad_;
};

std::shared_ptr<Layer> makeInput(Tensor *t, bool with_grad)
{
  return std::make_shared<Input>(t, with_grad);
}


}
