#include <math.h>
#include <assert.h>

#include <limits>
#include <sstream>
#include <algorithm>
#include <random>

#include "common.h"

namespace saga {

TensorStorage::~TensorStorage()
{
  chkCuda(cudaFree(device_mem_));
}





Tensor::Tensor(const Size &s, cudnnDataType_t data_type)
  : Size(s)
  , ns_(0)
  , cs_(0)
  , hs_(0)
  , ws_(0)
  , data_type_(data_type)
  , desc_(nullptr)
  , device_mem_(nullptr)
{}

Tensor::Tensor(const Size &s, cudnnDataType_t data_type, float v)
  : Tensor(s, data_type)
{
  fill(v);
}

Tensor::Tensor(const Tensor &t)
  : Tensor(t, t.dataType())
{
}

void Tensor::allocate()
{
  if(storage_)
    return; // Already allocated

  int wastedump;
  cudnnDataType_t wastedump2;
  chkCUDNN(cudnnCreateTensorDescriptor(&desc_));
  chkCUDNN(cudnnSetTensor4dDescriptor(desc_, CUDNN_TENSOR_NCHW, data_type_,
                                      n, c, h, w));

  cudnnGetTensor4dDescriptor(desc(), &wastedump2,
                             &wastedump, &wastedump, &wastedump, &wastedump,
                             &ns_, &cs_, &hs_, &ws_);

  size_t bytes;
  chkCUDNN(cudnnGetTensorSizeInBytes(desc(), &bytes));


  auto ts = std::make_shared<TensorStorage>();

  ts->bytes_ = bytes;
  chkCuda(cudaMallocManaged(&ts->device_mem_, bytes, cudaMemAttachGlobal));
  chkCuda(cudaMemset(ts->device_mem_, 0, bytes));
  storage_ = ts;
  device_mem_ = ts->device_mem_;
}


void Tensor::synchronize() const
{
  cudaDeviceSynchronize();
}




std::vector<unsigned int> Tensor::prediction() const
{
  synchronize();

  std::vector<unsigned int> r;
  r.reserve(n);

  for(unsigned int i = 0; i < n; i++) {
    unsigned int label = 0;
    float best = get(i, 0, 0, 0);
    for(unsigned int j = 1; j < c; j++) {
      float v = get(i, j, 0, 0);
      if(v > best) {
        label = j;
        best = v;
      }
    }
    r.push_back(label);
  }
  return r;
}


float Tensor::loss(const unsigned int *labels) const
{
  synchronize();

  float loss_sum = 0;
  for(unsigned int i = 0; i < n; i++) {
    float v = get(i, labels[i], 0, 0);
    loss_sum += -log(v);
  }
  return loss_sum / n;
}


void Tensor::load(const float *data)
{
  if(storage_ == NULL)
    allocate();
  memcpy(deviceMem(), (const void *)data, storage_->bytes_);
}


void Tensor::load(const std::vector<float> &data)
{
  load(&data[0]);
}

void Tensor::load(__restrict__ const uint8_t *data)
{
  const size_t num_elements = elements();
  float floats[num_elements];
  for(size_t i = 0; i < num_elements; i++)
    floats[i] = data[i] / 255.0f;
  load(floats);
}

void Tensor::load(__restrict__ const uint8_t **data)
{
  const size_t num_elements = elements();
  float floats[num_elements];
  const size_t chw = c * h * w;
  float *dst = &floats[0];
  for(size_t i = 0; i < n; i++) {
    const uint8_t *src = data[i];
    for(size_t j = 0; j < chw; j++)
      *dst++ = src[j] / 255.0f;
  }
  load(floats);
}


void Tensor::load(const void *data, size_t size)
{
  if(storage_ == NULL)
    allocate();
  assert(storage_->bytes_ == size);
  memcpy(deviceMem(), (const void *)data, storage_->bytes_);
}


void Tensor::fill(float value)
{
  assert(dataType() == CUDNN_DATA_FLOAT);

  if(value == 0) {
    if(storage_ == NULL)
      allocate();
    cudaMemset(deviceMem(), 0, storage_->bytes_);
    return;
  }

  load(std::vector<float>(elements(), value));
}


void Tensor::randomize(float sigma)
{
  assert(dataType() == CUDNN_DATA_FLOAT);

  if(sigma == 0) {
    fill(0);
    return;
  }

  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0,sigma);

  std::vector<float> values(elements());

  for(size_t i = 0; i < values.size(); i++) {
    values[i] = distribution(generator);
  }
  load(values);
}



std::string Size::name() const {
  std::stringstream ss;
  ss << "[" << n << ", " << c << ", " << w << ", " << h << "]";
  return ss.str();
}

void Tensor::dump(const char *prefix, bool intensity) const {

  const int dim_size = 4;
  const int in = n;
  const int ic = c;
  const int ih = h;
  const int iw = w;

  synchronize();

  for(int n = 0; n < in; n++) {
    if(in > dim_size * 2 && n == dim_size) n = in - dim_size;

    for(int c = 0; c < ic; c++) {
      if(ic > dim_size * 2 && c == dim_size) c = ic - dim_size;

      printf("%10s: N%-2dC%-3d", prefix, n, c);

      for(int y = 0; y < ih; y++) {
        if(ih > dim_size * 2 && y == dim_size) {
          y = ih - dim_size;
          printf("%10s: ...\n", prefix);
        }

        if(y) {
          printf("%10s:        ", "");
        }

        for(int x = 0; x < iw; x++) {
          if(iw > dim_size * 2 && x == dim_size) {
            x = iw - dim_size;
            printf(" ... ");
          }

          float v = get(n, c, y, x);
          if(intensity) {
            v = fabs(v);
            if(v < 0.25) {
              printf(" ");
            } else if(v < 0.5) {
              //              printf("%lc", 0x2591);
              printf(".");
            } else if(v < 0.75) {
              //              printf("%lc", 0x2592);
              printf("x");
            } else if(v < 2) {
              //              printf("%lc", 0x2593);
              printf("X");
            } else {
              //              printf("%lc", 0x2588);
              printf("#");
            }

          } else {
            printf("%s%2.6f ", v < 0 ? "" : " ", v);
          }
        }
        printf("\n");
      }
    }
  }
}

Tensor::Stats Tensor::stats() const {

  synchronize();

  float max = -INFINITY;
  float min = INFINITY;

  double sum = 0;

  for(unsigned int i = 0; i < n; i++) {
    for(unsigned int j = 0; j < c; j++) {
      for(unsigned int y = 0; y < h; y++) {
        for(unsigned int x = 0; x < w; x++) {
          float v = get(i, j, y, x);
          max = std::max(max, v);
          min = std::min(min, v);
          sum += v;
        }
      }
    }
  }

  const double mean = sum / elements();

  double sum2 = 0;

  for(unsigned int i = 0; i < n; i++) {
    for(unsigned int j = 0; j < c; j++) {
      for(unsigned int y = 0; y < h; y++) {
        for(unsigned int x = 0; x < w; x++) {
          float v = get(i, j, y, x) - mean;
          sum2 += v * v;
        }
      }
    }
  }

  Stats s;
  s.min = min;
  s.max = max;
  s.mean = mean;
  s.stddev = sqrt(sum2 / elements());
  return s;
}


}

