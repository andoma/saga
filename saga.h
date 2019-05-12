// -*-c++-*-

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>

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
  const int n, c, h, w;

  Size(int n, int c, int h, int w) : n(n), c(c), h(h), w(w) {};
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

private:

  void load(const TensorValues &v);

  void loadFloat(const float *src);

  void randomize(float sigma);

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

  virtual void backprop(const Network &n, const Tensor &dy) {};

  virtual void updateWeights(const Network &n) {};

  size_t workspaceSize() const { return workspace_size_; };

protected:

  Layer() : workspace_size_(0) {}

  size_t workspace_size_;
};



class Network {

public:
  Network(const Tensor &input, bool backprop);

  std::shared_ptr<Tensor> addLayer(std::shared_ptr<Layer> layer);

  void initialize();


  std::vector<std::shared_ptr<Layer>> layers_;

  cudnnHandle_t cudnn_;
  cublasHandle_t cublas_;

  int batch_size_;

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


}
