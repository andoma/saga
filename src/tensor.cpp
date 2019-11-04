#include <math.h>
#include <assert.h>

#include <limits>
#include <sstream>
#include <algorithm>
#include <random>

#include <x86intrin.h>

#include "common.h"

namespace saga {

static float
get_float(const void *base, size_t offset)
{
  return ((const float *)base)[offset];
}

static float
get_half(const void *base, size_t offset)
{
  return _cvtsh_ss(((const uint16_t *)base)[offset]);
}

static float
get_u8(const void *base, size_t offset)
{
  return ((const uint8_t *)base)[offset];
}

static void
set_float(void *base, size_t offset, float v)
{
  ((float *)base)[offset] = v;
}

static void
set_half(void *base, size_t offset, float v)
{
  ((uint16_t *)base)[offset] = _cvtss_sh(v, 0);
}

static void
set_u8(void *base, size_t offset, float v)
{
  ((uint8_t *)base)[offset] = v;
}




TensorStorage::~TensorStorage()
{
  chkCuda(cudaFree(device_mem_));
}

TensorStorage::TensorStorage(size_t bytes)
  : bytes_(bytes)
{
  chkCuda(cudaMallocManaged(&device_mem_, bytes_, cudaMemAttachGlobal));
  chkCuda(cudaMemset(device_mem_, 0, bytes_));
}


Tensor::~Tensor()
{
  cudnnDestroyTensorDescriptor(desc_);
}


Tensor::Tensor(const Size &s, Type type)
  : Size(s)
  , ns_(0)
  , cs_(0)
  , hs_(0)
  , ws_(0)
  , type_(type)
  , desc_(nullptr)
  , device_mem_(nullptr)
{
  switch(type) {
  case Type::FLOAT:
    element_size_ = sizeof(float);
    gettype_ = get_float;
    settype_ = set_float;
    break;

  case Type::HALF:
    element_size_ = 2;
    gettype_ = get_half;
    settype_ = set_half;
    break;

  case Type::U8:
    element_size_ = sizeof(uint8_t);
    gettype_ = get_u8;
    settype_ = set_u8;
    break;

  default:
    fprintf(stderr, "Unsupported data_type %d in tensor\n", (int)type);
    abort();
  }

  chkCUDNN(cudnnCreateTensorDescriptor(&desc_));
}

Tensor::Tensor(const Size &s, Type type, float v)
  : Tensor(s, type)
{
  fill(v);
}

Tensor::Tensor(const Tensor &t)
  : Tensor(t, t.type())
{
}

cudnnDataType_t
Tensor::cudnnType() const
{
  switch(type_) {
  case Tensor::Type::FLOAT:
    return CUDNN_DATA_FLOAT;
  case Tensor::Type::HALF:
    return CUDNN_DATA_HALF;
  case Tensor::Type::U8:
    return CUDNN_DATA_UINT8;
  default:
    fprintf(stderr, "Unsupported data_type %d in tensor\n", (int)type_);
    abort();
  }
}


void
Tensor::allocate(cudnnTensorFormat_t format)
{
  if(storage_)
    return; // Already allocated

  int wastedump;
  auto data_type = cudnnType();
  chkCUDNN(cudnnSetTensor4dDescriptor(desc_, format, data_type, n, c, h, w));

  cudnnGetTensor4dDescriptor(desc_, &data_type,
                             &wastedump, &wastedump, &wastedump, &wastedump,
                             &ns_, &cs_, &hs_, &ws_);

  size_t bytes;

  chkCUDNN(cudnnGetTensorSizeInBytes(desc_, &bytes));
  storage_ = std::make_shared<TensorStorage>(bytes);
  device_mem_ = storage_->device_mem_;
}


void
Tensor::allocate(Tensor *container, void *deviceMem)
{
  if(storage_)
    return; // Already allocated

  chkCUDNN(cudnnSetTensor4dDescriptorEx(desc_, cudnnType(),
                                        n, c, h, w,
                                        container->ns_,
                                        container->cs_,
                                        container->hs_,
                                        container->ws_));

  ns_ = container->ns_;
  cs_ = container->cs_;
  hs_ = container->hs_;
  ws_ = container->ws_;

  storage_ = container->storage_;
  device_mem_ = deviceMem;
}


void Tensor::synchronize() const
{
  cudaDeviceSynchronize();
}






void Tensor::load(const float *data)
{
  // Deprecate this
  assert(storage_ != NULL);
  memcpy(hostMem(), (const void *)data, storage_->bytes_);
}


void Tensor::load(const void *data, size_t size)
{
  // Deprecate this
  assert(storage_ != NULL);
  assert(storage_->bytes_ == size);
  memcpy(hostMem(), (const void *)data, storage_->bytes_);
}



void Tensor::load(const std::vector<float> &data)
{
  assert(type_ == Type::FLOAT);
  assert(storage_ != NULL);
  assert(storage_->bytes_ == data.size() * sizeof(float));

  memcpy(hostMem(), (const void *)&data[0], data.size() * sizeof(float));
}

void Tensor::load(const std::vector<uint16_t> &data)
{
  assert(type_ == Type::HALF);
  assert(storage_ != NULL);
  assert(storage_->bytes_ ==  data.size() * sizeof(uint16_t));

  memcpy(hostMem(), (const void *)&data[0], data.size() * sizeof(uint16_t));
}


void Tensor::load(__restrict__ const uint8_t **data)
{
  // This is a bit of an special case, doesn't belong here really
  // as it's only used from the mnist test code.
  // We should prolly do conversion as a layer on the GPU instead

  const size_t chw = c * h * w;

  switch(type_) {

  case Type::FLOAT: {
    std::vector<float> values(elements());
    size_t p = 0;
    for(size_t i = 0; i < n; i++) {
      const uint8_t *src = data[i];
      for(size_t j = 0; j < chw; j++)
        values[p++] = src[j] / 255.0f;
    }
    load(values);
    break;
  }

  case Type::HALF: {
    std::vector<uint16_t> values(elements());
    size_t p = 0;
    for(size_t i = 0; i < n; i++) {
      const uint8_t *src = data[i];
      for(size_t j = 0; j < chw; j++)
        values[p++] = _cvtss_sh(src[j] / 255.0f, 0);
    }
    load(values);
    break;
  }
  default:
    abort();
  }
}




void Tensor::fill(float value)
{
  allocate();

  if(value == 0) {
    cudaMemset(deviceMem(), 0, storage_->bytes_);
    return;
  }

  switch(type_) {
  case Type::FLOAT:
    load(std::vector<float>(elements(), value));
    break;
  case Type::HALF:
    load(std::vector<uint16_t>(elements(), _cvtss_sh(value, 0)));
    break;
  default:
    abort();
  }
}


void Tensor::randomize(float sigma)
{
  if(sigma == 0) {
    fill(0);
    return;
  }

  std::default_random_engine generator;
  std::normal_distribution<float> distribution(0,sigma);

  switch(type_) {
  case Type::FLOAT: {
    std::vector<float> values(elements());

    for(size_t i = 0; i < values.size(); i++) {
      values[i] = distribution(generator);
    }
    load(values);
    break;
  }

  case Type::HALF: {
    std::vector<uint16_t> values(elements());
    for(size_t i = 0; i < values.size(); i++) {
      values[i] = _cvtss_sh(distribution(generator), 0);
    }
    load(values);
    break;
  }

  default:
    abort();
  }
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

void Tensor::printStats(const char *postfix) const {
  const auto s = stats();
  printf("%f\t%f\t%f\t%f\t%s\n",
         s.min, s.max, s.mean, s.stddev, postfix);
}


void
printDesc(cudnnTensorDescriptor_t desc)
{
  cudnnDataType_t dt;
  int n, c, h, w;
  int ns, cs, hs, ws;

  cudnnGetTensor4dDescriptor(desc, &dt, &n, &c, &h, &w,
                             &ns, &cs, &hs, &ws);
  printf("dt:%d n=%d c=%d h=%d w=%d ns=%d cs=%d hs=%d ws=%d\n",
         dt,
         n, c, h, w,
         ns, cs, hs, ws);
}



}

