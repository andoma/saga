// -*-c++-*-

#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>

#include <assert.h>
#include <cudnn.h>
#include <cublas_v2.h>


namespace saga {

class Network;

enum class ActivationMode {
  RELU,
  ELU
};

enum class PoolingMode {
  MAX,
  AVERAGE
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

  Size(const std::vector<unsigned int> &v)
    : n(v.size() > 0 ? v[0] : 1)
    , c(v.size() > 1 ? v[1] : 1)
    , h(v.size() > 2 ? v[2] : 1)
    , w(v.size() > 3 ? v[3] : 1)
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
                   cudnnTensorFormat_t format,
                   unsigned int n,
                   unsigned int c,
                   unsigned int h,
                   unsigned int w);

  TensorDescriptor(cudnnDataType_t data_type,
                   cudnnTensorFormat_t format,
                   const Size &s)
    : TensorDescriptor(data_type, format, s.n, s.c, s.h, s.w)
  {}

  TensorDescriptor(TensorDescriptor const& td)
    : TensorDescriptor(td.data_type_, td.format_, td.n, td.c, td.h, td.w)
  {}

  ~TensorDescriptor();

  cudnnDataType_t dataType() const { return data_type_; }

  cudnnTensorFormat_t format() const { return format_; }

  cudnnTensorDescriptor_t desc() const { return desc_; }

  std::string name() const;

  Size strides() const;

private:
  cudnnDataType_t data_type_;
  cudnnTensorFormat_t format_;
  cudnnTensorDescriptor_t desc_;
};



class Tensor : public TensorDescriptor {

public:

  // Constructors

  static std::shared_ptr<Tensor> createFromPB(const char *path);

  Tensor& operator=(Tensor const&);

  Tensor(const TensorDescriptor &td);

  Tensor(Tensor const &t) : Tensor(TensorDescriptor(t)) {}

  ~Tensor();

  void *deviceMem(void) {
    if(device_mem_ == NULL)
      upload();
    return device_mem_;
  };

  void *deviceMem(void) const {
    assert(device_mem_ != NULL);
    return device_mem_;
  };

  void save(float *data) const;

  void savePng(const char *filename, int num = -1, int channel = -1) const;

  // Loaders

  void load(const std::vector<float> &data);

  void load(const uint8_t **data);

  void load(const float *data);

  void load(const uint8_t *data);

  void load(const void *data, size_t size);

  //  void load(const TensorValues &v);

  void randomize(float sigma);

  void fill(float value);

  // Helpers

  void dump(const char *prefix, bool intensity = false) const;

  void check() const;

  float peak() const;

  std::vector<unsigned int> prediction() const;

  float loss(const unsigned int *labels) const;


private:
  void *device_mem_;
  //  void *host_mem_;

  size_t bytes_;

  void upload(void);
};


class Layer {

public:

  virtual ~Layer() {};

  virtual std::string name() const = 0;

  virtual const Tensor *output() const = 0;

  virtual Tensor *gradient() const { return nullptr; }

  virtual void forward(const Network &n) = 0;

  virtual void backprop(const Network &n) {}

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

  bool load(const char *path);

  std::shared_ptr<Layer> addLayer(std::shared_ptr<Layer> layer);

  std::shared_ptr<Layer> nameLayer(std::shared_ptr<Layer> layer,
                                   const std::string &name);

  std::shared_ptr<Layer> findLayer(const std::string &name) const;

  void forward(bool inference);

  void backprop(unsigned int iteration);

  void setOptimizer(std::function<std::unique_ptr<Optimizer>(const Size &s,
                                                             const Network &net)> fn) {
    optimizer_factory_ = fn;
  }

  std::unique_ptr<Optimizer> makeOptimizer(const Size &s) const;

  std::vector<std::shared_ptr<Layer>> layers_;

  std::unordered_map<std::string, std::shared_ptr<Layer>> named_layers_;

  std::function<std::unique_ptr<Optimizer>(const Size &s,
                                           const Network &net)> optimizer_factory_;

  cudnnHandle_t cudnn_;
  cublasHandle_t cublas_;

  int batch_size_;

  bool backprop_;

  void *workspace_;
  size_t workspace_size_;

  int iteration_;

  bool inference_;
};


std::shared_ptr<Layer> makeFullyConnected(int num_outputs,
                                          const Layer &prev,
                                          const Network &n,
                                          std::shared_ptr<Tensor> weights = NULL,
                                          std::shared_ptr<Tensor> bias = NULL);

std::shared_ptr<Layer> makeConvolution(int activation_maps,
                                       int filter_size,
                                       int stride,
                                       int padding,
                                       const Layer &prev,
                                       const Network &n,
                                       bool with_bias = true,
                                       std::shared_ptr<Tensor> weights = NULL,
                                       std::shared_ptr<Tensor> bias = NULL);

std::shared_ptr<Layer> makeActivation(ActivationMode mode, float a,
                                      const Layer &prev,
                                      const Network &n);

std::shared_ptr<Layer> makePooling(PoolingMode mode,
                                   int size, int pad, int stride,
                                   const Layer &prev,
                                   const Network &n);

std::shared_ptr<Layer> makeSoftmax(const Layer &prev,
                                   const Network &n);

std::shared_ptr<Layer> makeDropout(float prob,
                                   std::shared_ptr<Layer> prev,
                                   const Network &n);

std::shared_ptr<Layer> makeInput(const Tensor *input,
                                 bool withGradient = false);

std::shared_ptr<Layer> makeConcat(const std::vector<const Layer *> &prevs,
                                  const Network &n);

std::shared_ptr<Layer> makeBatchNorm(double epsilon,
                                     const Layer &prev,
                                     const Network &n,
                                     std::shared_ptr<Tensor> scale,
                                     std::shared_ptr<Tensor> bias,
                                     std::shared_ptr<Tensor> mean,
                                     std::shared_ptr<Tensor> variance);

// Optimizers

std::unique_ptr<Optimizer> makeAdamOptimizer(const Size &s,
                                             const Network &net);

std::unique_ptr<Optimizer> makeGradientDescentOptimizer(const Size &s,
                                                        const Network &net);


}
