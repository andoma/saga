#include <dirent.h>
#include <math.h>
#include <assert.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>

#include "common.h"

using namespace std;

namespace saga {


static void
logcb(cudnnSeverity_t sev, void *udata, const cudnnDebug_t *dbg,
      const char *msg)
{
  fprintf(stderr, "%s\n", msg);
}

Network::Network(bool backprop)
  : backprop_(backprop)
  , workspace_(NULL)
  , workspace_size_(0)
  , setup_(false)
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

std::shared_ptr<Tensor> Network::findTensor(const char *name,
                                            const Size &s,
                                            Tensor::Type type,
                                            float mean,
                                            float sigma)
{
  if(name != NULL) {
    auto r = named_tensors_.find(name);
    if(r != named_tensors_.end()) {
      assert(*r->second == s);
      assert(r->second->type() == type);
      return r->second;
    }
  }

  auto t = make_shared<Tensor>(s, type);
  t->allocate(CUDNN_TENSOR_NHWC);
  if(sigma) {
    t->randomize(sigma);
  } else {
    t->fill(mean);
  }

  if(name != NULL)
    named_tensors_[name] = t;
  return t;
}

struct TensorDiskHeader {
  uint8_t magic[8];
  unsigned int n;
  unsigned int h;
  unsigned int w;
  unsigned int c;

} __attribute__((packed));



void Network::saveTensors(const char *path) const
{
  TensorDiskHeader tdh;
  memcpy(tdh.magic, "sagaT000", 8);

  mkdir(path, 0755);
  char filepath[PATH_MAX];
  for(const auto &it : named_tensors_) {
    snprintf(filepath, sizeof(filepath), "%s/%s", path, it.first.c_str());

    int fd = open(filepath, O_CREAT | O_TRUNC | O_RDWR, 0644);
    if(fd == -1) {
      fprintf(stderr, "Unable to create %s -- %s\n",
              filepath, strerror(errno));
      continue;
    }

    auto &t = it.second;
    tdh.n = t->n;
    tdh.h = t->h;
    tdh.w = t->w;
    tdh.c = t->c;
    if(write(fd, &tdh, sizeof(tdh)) != sizeof(tdh)) {
      fprintf(stderr, "Unable to write to %s\n",
              filepath);
      close(fd);
      unlink(filepath);
      continue;
    }

    ssize_t s = t->elements() * t->elementSize();

    if(write(fd, t->deviceMem(), s) != s) {
      fprintf(stderr, "Unable to write to %s\n",
              filepath);
      close(fd);
      unlink(filepath);
      continue;
    }
    close(fd);
  }
}

void Network::loadTensors(const char *path)
{
   struct dirent **namelist;
   int n = scandir(path, &namelist, NULL, NULL);
   if(n == -1) {
     fprintf(stderr, "Unable to load tensors from %s -- %s",
             path, strerror(errno));
     return;
   }


   while(n--) {
     const char *fname = namelist[n]->d_name;
     if(fname[0] != '.') {
       char filepath[PATH_MAX];
       snprintf(filepath, sizeof(filepath), "%s/%s", path, fname);

       int fd = open(filepath, O_RDONLY);
       if(fd != -1) {
         TensorDiskHeader tdh;

         if(read(fd, &tdh, sizeof(tdh)) == sizeof(tdh)) {
           if(!memcmp(tdh.magic, "sagaT000", 8)) {

             auto t = make_shared<Tensor>(Size(tdh.n, tdh.c, tdh.h, tdh.w),
                                          Tensor::Type::FLOAT);
             t->allocate(CUDNN_TENSOR_NHWC);
             ssize_t s = t->elements() * t->elementSize();
             if(read(fd, t->deviceMem(), s) != s) {
               fprintf(stderr, "Unable to read values from %s\n", path);
             } else {
               named_tensors_[fname] = t;
             }
           }

         } else {
           fprintf(stderr, "Unable to read header from %s\n", path);
         }
         close(fd);
       }
     }
     free(namelist[n]);
   }
   free(namelist);
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

  if(!setup_) {
    setup_ = true;
    auto last = layers_[layers_.size() - 1];

    last->output()->allocate();
    if(backprop_)
      last->gradient()->allocate();

    for(size_t i = 0; i < layers_.size(); i++) {
      layers_[i]->setup(*this);
      printf("Setup layer: %s\n", layers_[i]->name().c_str());
      workspace_size_ = std::max(workspace_size_, layers_[i]->workspaceSize());
    }
    chkCuda(cudaMalloc(&workspace_, workspace_size_));
    printf("workspace: %zd\n", workspace_size_);
  }

  for(size_t i = 0; i < layers_.size(); i++) {
    layers_[i]->forward(*this);
    if(layers_[i]->debug_)
      layers_[i]->output()->printStats(layers_[i]->name().c_str());
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
