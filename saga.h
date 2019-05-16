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

  Size(unsigned int n,
       unsigned int c,
       unsigned int h,
       unsigned int w)
    : n(n)
    , c(c)
    , h(h)
    , w(w)
  {}

  Size(const Size &s)
    : n(s.n)
    , c(s.c)
    , h(s.h)
    , w(s.w)
  {}

  size_t elements() const { return n * c * h * w; }

  inline bool operator==(const Size& o) const {
    return n == o.n && c == o.c && h == o.h && w == o.w;
  }

  std::string name() const;

};



struct TensorDescriptor : public Size {

  TensorDescriptor& operator=(TensorDescriptor const&) = delete;

  TensorDescriptor(cudnnDataType_t data_type,
                   unsigned int n,
                   unsigned int c,
                   unsigned int h,
                   unsigned int w);

  TensorDescriptor(cudnnDataType_t data_type, const Size &s)
    : TensorDescriptor(data_type, s.n, s.c, s.h, s.w)
  {}

  TensorDescriptor(TensorDescriptor const& td)
    : TensorDescriptor(td.data_type, td.n, td.c, td.h, td.w)
  {}

  ~TensorDescriptor();

  cudnnDataType_t dataType() const { return data_type; }

  cudnnTensorDescriptor_t desc() const { return desc_; }

  std::string name() const;

  cudnnDataType_t data_type; // move down to private:


private:
  cudnnTensorDescriptor_t desc_;
};





class TensorValues : public Size {

  TensorValues(const Size &s)
    : Size(s)
    , data_(elements())
  {}

  const std::vector<float> data_;

public:
  const float *data() const { return &data_[0]; }
};

typedef std::unordered_map<std::string, const TensorValues> InitData;




class Tensor : public TensorDescriptor {


public:

  Tensor& operator=(Tensor const&) = delete;

  Tensor(const TensorDescriptor &td);

  Tensor(Tensor const &t) : Tensor(TensorDescriptor(t)) {}


  ~Tensor();

  void loadOrRandomize(InitData id, const std::string &name, float sigma);

  void fill(float value);

  void *deviceMem(void) const { return device_mem_; };

  void save(float *data) const;

  void load(const std::vector<float> &data);

  void load(const uint8_t **data);

  void load(const float *data);

  void load(const uint8_t *data);

  void load(const TensorValues &v);

  void randomize(float sigma);

  void dump(const char *prefix, bool intensity = false) const;

  void check() const;

  float peak() const;

private:
  void *device_mem_;
  size_t bytes_;
};


class Layer {

public:

  virtual ~Layer() {};

  virtual std::string name() const = 0;

  virtual const Tensor *output() const = 0;

  virtual const Tensor *forward(const Network &n,
                                const Tensor &input,
                                bool inference) = 0;

  virtual const Tensor *backprop(const Network &n,
                                 const Tensor &input,
                                 const Tensor &grad,
                                 unsigned int iteration) {
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

  virtual void optimize(Tensor &x, const Tensor &grad, const Network &n,
                        unsigned int iternation) = 0;

};

typedef std::unique_ptr<Optimizer> (OptimizerFactory)(const Size &s,
                                                      const Network &net);

class Network {

public:
  Network(int batch_size, bool backprop);

  const Tensor *addLayer(std::shared_ptr<Layer> layer);

  void forward(const Tensor *input, bool inference);

  void backprop(const Tensor *input, const Tensor *dy,
                unsigned int iteration);

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

  bool backprop_;

  void *workspace_;
  size_t workspace_size_;
};


std::shared_ptr<Layer> makeFullyConnected(int num_outputs,
                                          const TensorDescriptor &input,
                                          const InitData &id,
                                          const Network &n);

std::shared_ptr<Layer> makeConvolution(int activation_maps,
                                       int filter_size,
                                       int stride,
                                       int padding,
                                       const TensorDescriptor &input,
                                       const InitData &id,
                                       const Network &n);

std::shared_ptr<Layer> makeActivation(ActivationMode mode, float a,
                                      const TensorDescriptor &input,
                                      const Network &n);

std::shared_ptr<Layer> makePooling(PoolingMode mode, int size, int stride,
                                   const TensorDescriptor &input,
                                   const Network &n);

std::shared_ptr<Layer> makeSoftmax(const TensorDescriptor &input,
                                   const Network &n);


// Optimizers

std::unique_ptr<Optimizer> makeAdamOptimizer(const Size &s,
                                             const Network &net);

std::unique_ptr<Optimizer> makeGradientDescentOptimizer(const Size &s,
                                                        const Network &net);


}
