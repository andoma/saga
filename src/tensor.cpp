#include <math.h>
#include <assert.h>

#include <limits>
#include <sstream>

#include "common.h"

namespace saga {



// https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
static double
generateGaussianNoise(void)
{
  static __thread double z1;
  static __thread bool generate;
  static const double epsilon = std::numeric_limits<double>::min();

  generate = !generate;

  if (!generate)
    return z1;

  double u1, u2;
  do {
    u1 = drand48();
    u2 = drand48();
  } while(u1 <= epsilon);

  double z0;
  z0 = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
  z1 = sqrt(-2.0 * log(u1)) * sin(2.0 * M_PI * u2);
  return z0;
}

Tensor::Tensor(cudnnDataType_t data_type, const Size &s)
  : data_type_(data_type)
  , size_(s)
  , device_mem_(NULL)
{
  chkCUDNN(cudnnCreateTensorDescriptor(&desc_));
  chkCUDNN(cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW,
                                      data_type, s.n, s.c, s.h, s.w));

  chkCUDNN(cudnnGetTensorSizeInBytes(desc_, &bytes_));
  chkCuda(cudaMalloc(&device_mem_, bytes_));
  chkCuda(cudaMemset(device_mem_, 0, bytes_));
};

Tensor::~Tensor()
{
  chkCUDNN(cudnnDestroyTensorDescriptor(desc_));
}



void Tensor::fill(float value)
{
  assert(data_type_ == CUDNN_DATA_FLOAT);

  const size_t bytes = size_.elements() * sizeof(float);

  if(value == 0) {
    cudaMemset(device_mem_, 0, bytes);
    return;
  }

  float *hmem = (float *)malloc(bytes);

  for(size_t i = 0; i < size_.elements(); i++) {
    hmem[i] = value;
  }
  cudaMemcpy(device_mem_, hmem, bytes, cudaMemcpyHostToDevice);
  free(hmem);
}


void Tensor::randomize(float sigma)
{
  assert(data_type_ == CUDNN_DATA_FLOAT);

  if(sigma == 0) {
    fill(0);
    return;
  }

  const size_t bytes = size_.elements() * sizeof(float);

  float *hmem = (float *)malloc(bytes);

  for(size_t i = 0; i < size_.elements(); i++) {
    hmem[i] = generateGaussianNoise() * sigma;
  }
  cudaMemcpy(device_mem_, hmem, bytes, cudaMemcpyHostToDevice);
  free(hmem);
}


void Tensor::load(const TensorValues &v)
{
  assert(v.size() == size());

  const size_t bytes = v.size().elements() * sizeof(float);
  cudaMemcpy(device_mem_, v.data(), bytes, cudaMemcpyHostToDevice);
}


void Tensor::loadOrRandomize(InitData id, const std::string &name, float sigma)
{
  auto it = id.find(name);
  if(it != id.end()) {
    load(it->second);
  } else {
    randomize(sigma);
  }
}


std::string Size::name() const {
  std::stringstream ss;
  ss << "[" << n << ", " << c << ", " << w << ", " << h << "]";
  return ss.str();

}


std::string Tensor::name() const {
  return size().name();
}


}
