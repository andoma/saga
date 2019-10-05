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


class TensorStorage {

public:

  TensorStorage(size_t bytes);

  ~TensorStorage();

  void *device_mem_;
  size_t bytes_;
};


class Tensor : public Size {

  struct Stats {
    float min;
    float max;
    float mean;
    float stddev;
  };

public:

  // Constructors

  static std::shared_ptr<Tensor> createFromPB(const char *path);


  explicit Tensor(const Tensor &t);

  explicit Tensor(const Size &s, cudnnDataType_t dt);

  explicit Tensor(const Size &s, cudnnDataType_t data_type, float fill_value);

  virtual ~Tensor();

  virtual void allocate(cudnnTensorFormat_t format = CUDNN_TENSOR_NHWC);

  void allocate(Tensor *container, size_t offset);

  cudnnDataType_t dataType() const { return data_type_; }

  cudnnTensorDescriptor_t desc() const {
    assert(desc_ != NULL);
    return desc_;
  }

  void *deviceMem(void) const {
    assert(device_mem_ != NULL);
    return device_mem_;
  };

  void save(float *data) const;

  void toRGBBitmap(uint8_t *output, int stride,
                   int num, int channel = -1,
                   float min = 0, float max = 1.0f) const;

  // Loaders

  void load(const std::vector<float> &data);

  void load(const uint8_t **data);

  void load(const float *data);

  void load(const uint8_t *data);

  void load(const void *data, size_t size);

  void randomize(float sigma);

  void fill(float value);

  // Helpers

  void dump(const char *prefix, bool intensity = false) const;

  void check() const;

  struct Stats stats() const;

  void printStats(const char *postfix) const;

  std::vector<unsigned int> prediction() const;

  float loss(const unsigned int *labels) const;

  void synchronize() const;

  size_t elementSize() const { return element_size_; }

  float get(int n, int c, int x, int y) const {
    const size_t o = n * ns_ + c * cs_ + y * hs_ + x * ws_;
    switch(data_type_) {
    case CUDNN_DATA_FLOAT:
      return ((const float *)device_mem_)[o];
    case CUDNN_DATA_UINT8:
      return ((const uint8_t *)device_mem_)[o];
    default:
      abort();
    }
  }

  void *getAddr(int n, int c, int x, int y) {
    char *p = (char *)device_mem_;
    return p + (n * ns_ + c * cs_ + y * hs_ + x * ws_) * element_size_;
  }

  void set(int n, int c, int x, int y, float v) {
    float *p = (float *)device_mem_;
    p[n * ns_ + c * cs_ + y * hs_ + x * ws_] = v;
  }


  int ns_;
  int cs_;
  int hs_;
  int ws_;

  std::shared_ptr<TensorStorage> storage_;

  cudnnTensorDescriptor_t desc_;

private:
  cudnnDataType_t data_type_;

  void *device_mem_;

  size_t element_size_;

};


class Layer {

public:

  virtual ~Layer() {};

  virtual std::string name() const = 0;

  virtual Tensor *output() const = 0;

  virtual Tensor *gradient() const { return nullptr; }

  virtual void setup(const Network &n) {};

  virtual void forward(const Network &n) = 0;

  virtual void backprop(const Network &n) {}

  virtual float loss() const { return NAN; }

  size_t workspaceSize() const { return workspace_size_; };

protected:

  Layer() : workspace_size_(0), debug_(false) {}

  size_t workspace_size_;
public:
  bool debug_;
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

  std::shared_ptr<Tensor> findTensor(const char *name,
                                     const Size &s,
                                     cudnnDataType_t dt,
                                     float mean,
                                     float sigma);

  std::unique_ptr<Optimizer> makeOptimizer(const Size &s) const;

  void saveTensors(const char *path) const;

  void loadTensors(const char *path);

  std::vector<std::shared_ptr<Layer>> layers_;

  std::unordered_map<std::string, std::shared_ptr<Layer>> named_layers_;

  std::unordered_map<std::string, std::shared_ptr<Tensor>> named_tensors_;

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
                                          Network &n,
                                          const char *weights = NULL,
                                          const char *bias = NULL);

std::shared_ptr<Layer> makeConvolution(int activation_maps,
                                       int filter_size,
                                       int stride,
                                       int padding,
                                       const Layer &prev,
                                       Network &n,
                                       bool with_bias = true,
                                       const char *weights = NULL,
                                       const char *bias = NULL);

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

std::shared_ptr<Layer> makeInput(Tensor *input,
                                 bool withGradient = false);

std::shared_ptr<Layer> makeConcat(const std::vector<const Layer *> &prevs,
                                  const Network &n);

std::shared_ptr<Layer> makeBatchNorm(double epsilon,
                                     const Layer &prev,
                                     Network &n,
                                     float expavgf,
                                     const char *scale = NULL,
                                     const char *bias = NULL,
                                     const char *mean = NULL,
                                     const char *var = NULL);

std::shared_ptr<Layer> makeMathOp(const Layer &prev,
                                  cudnnOpTensorOp_t op,
                                  std::shared_ptr<Tensor> b,
                                  float alpha1,
                                  float alpha2,
                                  Network &net);


std::shared_ptr<Layer> makeCatClassifier(const Layer &prev,
                                         cudnnDataType_t data_type,
                                         const Network &n);

// Optimizers

std::unique_ptr<Optimizer> makeAdamOptimizer(const Size &s,
                                             const Network &net,
                                             float learning_rate);

std::unique_ptr<Optimizer> makeGradientDescentOptimizer(const Size &s,
                                                        const Network &net);


// Misc helpers

void printDesc(cudnnTensorDescriptor_t desc);

}
