/*
 * Copyright (c) 2019, Andreas Smas
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#if 0

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
  , hostmustsync_(false)
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

void *Tensor::hostMem() const
{
  if(hostmustsync_) {
    cudaDeviceSynchronize();
    hostmustsync_ = false;
  }

  return device_mem_;
}



void Tensor::load(const std::vector<float> &data)
{
  assert(type_ == Type::FLOAT);
  assert(storage_ != NULL);
  assert(storage_->bytes_ == data.size() * sizeof(float));

  memcpy(hostMem(), (const void *)&data[0],
         data.size() * sizeof(float));
}

void Tensor::load(const std::vector<uint16_t> &data)
{
  assert(type_ == Type::HALF);
  assert(storage_ != NULL);
  assert(storage_->bytes_ ==  data.size() * sizeof(uint16_t));

  memcpy(hostMem(), (const void *)&data[0],
         data.size() * sizeof(uint16_t));
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


void Tensor::copyFrom(const Tensor &src)
{
  assert((Size)src == (Size)*this);
  assert(src.ns_ == ns_);
  assert(src.cs_ == cs_);
  assert(src.hs_ == hs_);
  assert(src.ws_ == ws_);
  assert(src.storage_->bytes_ == storage_->bytes_);

  cudaMemcpy(this->deviceMem(),
             src.deviceMem(),
             storage_->bytes_,
             cudaMemcpyDeviceToDevice);
}


float Tensor::compare(const Tensor &src)
{
  assert((Size)src == (Size)*this);

  float maxdiff = 0;

  for(size_t i = 0; i < n; i++) {
    for(size_t y = 0; y < h; y++) {
      for(size_t x = 0; x < w; x++) {
        for(size_t j = 0; j < c; j++) {
          const float d = fabs(get(i,j,y,x) - src.get(i,j,y,x));
          maxdiff = std::max(maxdiff, d);
        }
      }
    }
  }
  return maxdiff;
}



std::string Size::name() const {
  std::stringstream ss;
  ss << "[" << n << ", " << c << ", " << h << ", " << w << "]";
  return ss.str();
}

void Tensor::dump(const char *prefix, bool intensity) const {

  const int dim_size = 4;
  const int in = n;
  const int ic = c;
  const int ih = h;
  const int iw = w;

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

#endif

#include <sstream>
#include <x86intrin.h>

#include "saga.h"

namespace saga {

static double
get_float(const void *base, size_t offset)
{
  return ((const float *)base)[offset];
}

static double
get_half(const void *base, size_t offset)
{
  return _cvtsh_ss(((const uint16_t *)base)[offset]);
}

static double
get_u8(const void *base, size_t offset)
{
  return ((const uint8_t *)base)[offset];
}

static double
get_i64(const void *base, size_t offset)
{
  return ((const int64_t *)base)[offset];
}

static void
set_float(void *base, size_t offset, double v)
{
  ((float *)base)[offset] = v;
}

static void
set_half(void *base, size_t offset, double v)
{
  ((uint16_t *)base)[offset] = _cvtss_sh(v, 0);
}

static void
set_u8(void *base, size_t offset, double v)
{
  ((uint8_t *)base)[offset] = v;
}

static void
set_i64(void *base, size_t offset, double v)
{
  ((int64_t *)base)[offset] = v;
}

typedef double (getfn_t)(const void *base, size_t offset);
typedef void (setfn_t)(void *base, size_t offset, double value);


const getfn_t *
datatype_get(Tensor::DataType dt)
{
  switch(dt) {
  case Tensor::DataType::U8:    return &get_u8;
  case Tensor::DataType::HALF:  return &get_half;
  case Tensor::DataType::FLOAT: return &get_float;
  case Tensor::DataType::INT64: return &get_i64;
  default: abort();
  }
}

const setfn_t *
datatype_set(Tensor::DataType dt)
{
  switch(dt) {
  case Tensor::DataType::U8:    return &set_u8;
  case Tensor::DataType::HALF:  return &set_half;
  case Tensor::DataType::FLOAT: return &set_float;
  case Tensor::DataType::INT64: return &set_i64;
  default: abort();
  }
}


const char *
datatype_str(Tensor::DataType dt)
{
  switch(dt) {
  case Tensor::DataType::U8:    return "u8";
  case Tensor::DataType::HALF:  return "half";
  case Tensor::DataType::FLOAT: return "float";
  case Tensor::DataType::INT64: return "i64";
  default: return "?";
  }
}

const size_t
datatype_size(Tensor::DataType dt)
{
  switch(dt) {
  case Tensor::DataType::U8:    return 1;
  case Tensor::DataType::HALF:  return 2;
  case Tensor::DataType::FLOAT: return 4;
  case Tensor::DataType::INT64: return 8;
  default: abort();
  }
}



std::string
Tensor::info() const
{
  std::stringstream ss;
  ss << "\"" << name_ << "\"";
  ss << "<" << datatype_str(data_type_) << ">";
  const char *prefix = "[";
  for(const auto &x : dims_) {
    ss << prefix << x;
    prefix = ", ";
  }
  ss << "]";
  return ss.str();
}


static void
print1dTensor(const char *prefix, Tensor &t, TensorAccess &ta)
{
  for(int64_t i = 0; i < t.dims_[0]; i++) {
    printf("%s: [%5d]: %f\n", prefix, (int)i, ta.get({i}));
  }
}


static void
print2dTensor(const char *prefix, Tensor &t, TensorAccess &ta)
{
  for(int64_t y = 0; y < t.dims_[0]; y++) {
    printf("%s: [%5d]: ", prefix, (int)y);
    for(int64_t x = 0; x < t.dims_[1]; x++) {
      printf("% 3.3f", ta.get({y, x}));
    }
    printf("\n");
  }
}


void
Tensor::print(const char *prefix) {

  printf("%s: %s\n", prefix, info().c_str());

  auto ta = access();
  if(ta == nullptr) {
    printf("%s: Abstract (no data)\n", prefix);
    return;
  }

  if(dims_.size() == 1) {
    print1dTensor(prefix, *this, *ta);
    return;
  } else if(dims_.size() == 2) {
    print2dTensor(prefix, *this, *ta);
    return;
  }

  printf("%s: Can't print n-dimensional tensors yet\n", prefix);
}


std::unique_ptr<TensorAccess>
Tensor::access()
{
  return nullptr;
}



static Dims
computeCPUStrides(const Dims &dims)
{
  Dims strides;
  int stride = 1;
  for(int i = dims.size() - 1; i >= 0; i--) {
    strides.insert(strides.begin(), stride);
    stride *= dims[i];
  }
  return strides;
}



class TensorStorage {

public:

  TensorStorage(Tensor::DataType data_type, size_t elements)
    : data_type_(data_type)
    , elements_(elements)
    , data_type_size_(datatype_size(data_type))
    , data_size_(elements * data_type_size_)
    , get_(datatype_get(data_type))
    , set_(datatype_set(data_type))
    , data_(NULL)
  {}

  double get(size_t offset) const {
    return get_(data_, offset);
  }

  void set(size_t offset, double value) {
    set_(data_, offset, value);
  }

  const Tensor::DataType data_type_;
  const size_t elements_;
  const size_t data_type_size_;
  const size_t data_size_;
  const getfn_t *get_;
  const setfn_t *set_;
  void *data_;
};




class CPUTensorStorage : public TensorStorage {

public:
  CPUTensorStorage(Tensor::DataType data_type, const Dims &dims, const Dims &strides)
    : TensorStorage(data_type, dims[0] * strides[0])
  {
    data_ = calloc(1, data_size_);
  }

  ~CPUTensorStorage()
  {
    free(data_);
  }
};



class CPUTensorAccess : public TensorAccess {

public:
  CPUTensorAccess(const Dims &strides, std::shared_ptr<CPUTensorStorage> storage)
    : strides_(strides)
    , storage_(storage)
  {}

  ~CPUTensorAccess() {}

  Dims strides() { return strides_; }

  void *data() { return storage_->data_; }

  size_t offsetForElement(const std::vector<int64_t> &element) const {
    size_t offset = 0;
    for(size_t i = 0; i < element.size() && i < strides_.size(); i++) {
      offset += element[i] * strides_[i];
    }
    return offset;
  }

  virtual double get(const std::vector<int64_t> &element) const {
    return storage_->get(offsetForElement(element));
  };

  virtual void set(const std::vector<int64_t> &element, double value) {
    storage_->set(offsetForElement(element), value);
  }

  const Dims strides_;
  const std::shared_ptr<CPUTensorStorage> storage_;
};



class CPUTensor : public Tensor {
public:
  CPUTensor(const std::string &name, DataType data_type, Dims dims)
    : Tensor(name, data_type, dims)
    , strides_(computeCPUStrides(dims))
    , storage_(std::make_shared<CPUTensorStorage>(data_type, dims, strides_))
  {}

  std::unique_ptr<TensorAccess> access() {
    return std::make_unique<CPUTensorAccess>(strides_, storage_);
  }

  virtual std::string info() const;

  const Dims strides_;
  const std::shared_ptr<CPUTensorStorage> storage_;
};


std::string
CPUTensor::info() const
{
  std::stringstream ss;

  ss << Tensor::info();
  const char *prefix = "{";
  for(const auto &x : strides_) {
    ss << prefix << x;
    prefix = ", ";
  }
  ss << "}";
  return ss.str();
}



std::shared_ptr<Tensor>
makeCPUTensor(const std::string &name, Tensor::DataType data_type, Dims dims)
{
  return std::make_shared<CPUTensor>(name, data_type, dims);
}


}



