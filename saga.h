// -*-c++-*-

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>

#include <cudnn.h>
#include <cublas_v2.h>


namespace saga {

class Network;

enum class ActivationMode {
  RELU,
  ELU
};

enum class PoolingMode {
  MAX
};




struct Size {
  const unsigned int n, c, h, w;

  Size(unsigned int n, unsigned int c,
       unsigned int h, unsigned int w) : n(n), c(c), h(h), w(w) {};
  size_t elements() const { return n * c * h * w; };

  std::string name() const;

  inline bool operator==(const Size& o) {
    return n == o.n && c == o.c && h == o.h && w == o.w;
  }
};



class TensorValues {
  TensorValues(const Size &s)
    : size_(s)
    , data_(s.elements())
  {}

  const Size size_;
  const std::vector<float> data_;

public:
  Size size() const { return size_; };
  const float *data() const { return &data_[0]; }
};

typedef std::unordered_map<std::string, const TensorValues> InitData;




class Tensor {

public:
  Tensor(cudnnDataType_t data_type, const Size &s);
  Tensor(const Tensor &t) : Tensor(t.data_type_, t.size_) {}

  ~Tensor();

  std::string name() const;

  Size size() const { return size_; };

  cudnnDataType_t dataType() const { return data_type_; }

  void loadOrRandomize(InitData id, const std::string &name, float sigma);

  void fill(float value);

  void *deviceMem(void) const { return device_mem_; };

  cudnnTensorDescriptor_t desc() const { return desc_; }

  static std::shared_ptr<Tensor> make(cudnnDataType_t data_type,
                                      const Size &s) {
    return std::make_shared<Tensor>(data_type, s);
  }

  static std::shared_ptr<Tensor> make(const Tensor &blueprint) {
    return std::make_shared<Tensor>(blueprint.data_type_,
                                    blueprint.size_);
  }

  void save(float *data);

  void load(const std::vector<float> &data);

  void load(const uint8_t **data);

  void load(const float *data);

  void load(const uint8_t *data);

  void load(const TensorValues &v);

  void randomize(float sigma);

  void dump(const char *prefix, bool intensity) const;

  void check() const;

  float peak() const;

private:

  const cudnnDataType_t data_type_;
  const Size size_;
  size_t bytes_;
  cudnnTensorDescriptor_t desc_;
  void *device_mem_;
};


class Layer {

public:

  virtual ~Layer() {};

  virtual std::string name() const = 0;

  virtual std::shared_ptr<Tensor> output() const = 0;

  virtual void forward(const Network &n) = 0;

  virtual std::shared_ptr<Tensor> backprop(const Network &n,
                                           const Tensor &dy) {
    return nullptr;
  }

  size_t workspaceSize() const { return workspace_size_; };

protected:

  Layer() : workspace_size_(0) {}

  size_t workspace_size_;
};



class Optimizer {

public:

  virtual ~Optimizer() {};

  virtual void optimize(Tensor &x, const Tensor &grad, const Network &n) = 0;

};

typedef std::unique_ptr<Optimizer> (OptimizerFactory)(const Size &s,
                                                      const Network &net);

class Network {

public:
  Network(int batch_size, bool backprop);

  std::shared_ptr<Tensor> addLayer(std::shared_ptr<Layer> layer);

  void forward();

  void backprop(std::shared_ptr<Tensor> dy);

  void setOptimizer(std::function<std::unique_ptr<Optimizer>(const Size &s,
                                                             const Network &net)> fn) {
    optimizer_factory_ = fn;
  }

  std::unique_ptr<Optimizer> makeOptimizer(const Size &s) const;

  std::vector<std::shared_ptr<Layer>> layers_;

  std::function<std::unique_ptr<Optimizer>(const Size &s,
                                           const Network &net)> optimizer_factory_;

  cudnnHandle_t cudnn_;
  cublasHandle_t cublas_;

  int batch_size_;
  int iter_;

  bool backprop_;
  bool inference_;

  void *workspace_;
  size_t workspace_size_;
};


std::shared_ptr<Layer> makeFullyConnected(int num_outputs,
                                          std::shared_ptr<Tensor> input,
                                          const InitData &id,
                                          const Network &n);

std::shared_ptr<Layer> makeConvolution(int activation_maps,
                                       int filter_size,
                                       int stride,
                                       int padding,
                                       std::shared_ptr<Tensor> input,
                                       const InitData &id,
                                       const Network &n);

std::shared_ptr<Layer> makeActivation(ActivationMode mode, float a,
                                      std::shared_ptr<Tensor> input,
                                      const Network &n);

std::shared_ptr<Layer> makePooling(PoolingMode mode, int size, int stride,
                                   std::shared_ptr<Tensor> input,
                                   const Network &n);

std::shared_ptr<Layer> makeSoftmax(std::shared_ptr<Tensor> input,
                                   const Network &n);


// Optimizers

std::unique_ptr<Optimizer> makeAdamOptimizer(const Size &s,
                                             const Network &net);

std::unique_ptr<Optimizer> makeGradientDescentOptimizer(const Size &s,
                                                        const Network &net);


}
