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


#include <random>
#include <sstream>

#include <inttypes.h>
#include <x86intrin.h>
#include <math.h>


#include "saga.h"
#include "tensor.h"

namespace saga {

//------------------------------------------------------------------------

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

static double
get_i32(const void *base, size_t offset)
{
  return ((const int32_t *)base)[offset];
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

static void
set_i32(void *base, size_t offset, double v)
{
  ((int32_t *)base)[offset] = v;
}

const TensorStorageAccess::getfn_t *
datatype_get(Tensor::DataType dt)
{
  switch(dt) {
  case Tensor::DataType::U8:    return &get_u8;
  case Tensor::DataType::HALF:  return &get_half;
  case Tensor::DataType::FLOAT: return &get_float;
  case Tensor::DataType::INT64: return &get_i64;
  case Tensor::DataType::I32:   return &get_i32;
  default: abort();
  }
}

const TensorStorageAccess::setfn_t *
datatype_set(Tensor::DataType dt)
{
  switch(dt) {
  case Tensor::DataType::U8:    return &set_u8;
  case Tensor::DataType::HALF:  return &set_half;
  case Tensor::DataType::FLOAT: return &set_float;
  case Tensor::DataType::INT64: return &set_i64;
  case Tensor::DataType::I32:   return &set_i32;
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
  case Tensor::DataType::I32:   return "i32";
  default: return "?";
  }
}

size_t
Tensor::DataTypeSize(DataType dt)
{
  switch(dt) {
  case Tensor::DataType::U8:    return 1;
  case Tensor::DataType::HALF:  return 2;
  case Tensor::DataType::FLOAT: return 4;
  case Tensor::DataType::INT64: return 8;
  case Tensor::DataType::I32:   return 4;
  default: abort();
  }
}




TensorStorageAccess::TensorStorageAccess(Tensor::DataType data_type)
  : get_(datatype_get(data_type))
  , set_(datatype_set(data_type))
  , data_type_(data_type)
  , data_(NULL)
{}


//------------------------------------------------------------------------


int64_t
elements_from_dims(const Dims &dims)
{
  int64_t elements = 1;
  for(auto &d : dims)
    elements *= d;
  return elements;
}



Tensor::Tensor(DataType data_type, const Dims &dims,
               const std::optional<const std::string> &name)
  : name_(name)
  , data_type_(data_type)
  , dims_(dims)
  , elements_(elements_from_dims(dims))
{};



std::string
Tensor::info() const
{
  std::stringstream ss;
  if(name_) {
    ss << "\"" << *name_ << "\"";
  }
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


Tensor::Stats
Tensor::stats()
{
  auto ta = access();
  if(ta == nullptr) {
    return Stats({});
  }
  std::vector<int64_t> c(dims_.size(), 0);

  double max = -INFINITY;
  double min = INFINITY;
  double sum = 0;

  for(int64_t i = 0; i < elements_; i++) {
    const double v = ta->get(c);

    max = std::max(max, v);
    min = std::min(min, v);
    sum += v;

    for(ssize_t j = c.size() - 1; j >= 0; j--) {
      c[j]++;
      if(c[j] == dims_[j]) {
        c[j] = 0;
      } else {
        break;
      }
    }
  }

  const double mean = sum / elements_;
  double sum2 = 0;
  for(int64_t i = 0; i < elements_; i++) {
    const double v = ta->get(c) - mean;
    sum2 += v * v;
    for(ssize_t j = c.size() - 1; j >= 0; j--) {
      c[j]++;
      if(c[j] == dims_[j]) {
        c[j] = 0;
      } else {
        break;
      }
    }
  }
  return Stats({.min = min, .max = max, .mean = mean,
        .stddev = sqrt(sum2 / elements_)});
}


std::string
Tensor::statsString(void)
{
  auto s = stats();
  char buf[512];
  snprintf(buf, sizeof(buf), "{min:%f mean:%f max:%f stddev:%f}",
           s.min, s.mean, s.max, s.stddev);
  return std::string(buf);
}


void
Tensor::printStats(const char *prefix)
{
  auto s = stats();
  printf("%s: min:%f max:%f mean:%f stddev:%f\n", prefix,
         s.min, s.max, s.mean, s.stddev);
}


void
Tensor::print(const char *prefix, int elements_per_rank)
{
  printf("%s: %s\n", prefix, info().c_str());

  auto ta = access();
  if(ta == nullptr) {
    printf("%s: Abstract (no data)\n", prefix);
    return;
  }

  if(dims_.size() == 1) {
    // We format 1d tensor vertically instead of a long horizontal line
    print1dTensor(prefix, *this, *ta);
    return;
  }

  const size_t rank = dims_.size();
  std::vector<int64_t> c(rank, 0);

  const char *lf = "";

  while(1) {
    if(c[rank - 1] == 0) {
      printf("%s%s: [", lf, prefix);
      for(size_t j = 0; j < c.size(); j++) {
        printf("%s%3" PRId64, j ? "," : "", c[j]);
      }
      printf("]");
      lf = "\n";
    }
    printf(" % 3.3f", ta->get(c));

    for(ssize_t j = rank - 1; j >= 0; j--) {
      c[j]++;
      if(c[j] == dims_[j] || c[j] == elements_per_rank) {
        if(j == 0) {
          printf("%s", lf);
          return;
        }
        c[j] = 0;
      } else {
        break;
      }
    }
  }
}



void
Tensor::printRGB(const char *prefix, float scale)
{
  printf("%s: %s\n", prefix, info().c_str());

  auto ta = access();
  if(ta == nullptr) {
    printf("%s: Abstract (no data)\n", prefix);
    return;
  }

  if(dims_.size() < 3) {
    // We format 1d tensor vertically instead of a long horizontal line
    printf("%s: Too few dimensions\n", prefix);
    return;
  }

  size_t dim_offset = dims_.size() - 3;

  for(int n = 0; n < dims_[0]; n++) {

    for(int y = 0; y < dims_[dim_offset + 1]; y++) {
      printf("%s: [%d]", prefix, n);

      if(dims_[dim_offset] == 1) {
        for(int x = 0; x < dims_[dim_offset + 2]; x++) {
          const int v = ta->get({n,0,y,x}) * scale;
          printf("\033[48;2;%d;%d;%dm ", v, v, v);
        }
      } else {
        for(int x = 0; x < dims_[dim_offset + 2]; x++) {
          const int r = ta->get({n,0,y,x}) * scale;
          const int g = ta->get({n,1,y,x}) * scale;
          const int b = ta->get({n,2,y,x}) * scale;
          printf("\033[48;2;%d;%d;%dm ", r, g, b);
        }
      }
      printf("\033[0m\n");
    }
  }
}


void
Tensor::copyFrom(Tensor &t)
{
  auto src = t.access();
  if(src == nullptr)
    return;

  auto dst = access();

  assert(t.elements_ == elements_);

  std::vector<int64_t> c_s(t.dims_.size(), 0);
  std::vector<int64_t> c_d(dims_.size(), 0);

  // This is very slow
  for(int64_t i = 0; i < elements_; i++) {
    dst->set(c_d, src->get(c_s));

    for(ssize_t j = c_d.size() - 1; j >= 0; j--) {
      c_d[j]++;
      if(c_d[j] == dims_[j]) {
        c_d[j] = 0;
      } else {
        break;
      }
    }

    for(ssize_t j = c_s.size() - 1; j >= 0; j--) {
      c_s[j]++;
      if(c_s[j] == t.dims_[j]) {
        c_s[j] = 0;
      } else {
        break;
      }
    }
  }
}


double
Tensor::sse(Tensor &t)
{
  auto a = t.access();
  auto b = access();
  if(a == nullptr && b == nullptr)
    return 0.0;

  if(a == nullptr || b == nullptr)
    return INFINITY;

  assert(t.elements_ == elements_);

  std::vector<int64_t> c_a(t.dims_.size(), 0);
  std::vector<int64_t> c_b(dims_.size(), 0);

  double r = 0;
  for(int64_t i = 0; i < elements_; i++) {

    double v = a->get(c_a) - b->get(c_b);
    r += v * v;

    for(ssize_t j = c_a.size() - 1; j >= 0; j--) {
      c_a[j]++;
      if(c_a[j] == dims_[j]) {
        c_a[j] = 0;
      } else {
        break;
      }
    }

    for(ssize_t j = c_b.size() - 1; j >= 0; j--) {
      c_b[j]++;
      if(c_b[j] == t.dims_[j]) {
        c_b[j] = 0;
      } else {
        break;
      }
    }
  }
  return r;
}


std::optional<const std::string>
Tensor::namePostfix(const std::string &postfix) const
{
  return name_ ? std::make_optional(*name_ + "." + postfix) : std::nullopt;
}




class GenTensorAccess : public TensorAccess {

public:
  GenTensorAccess(size_t rank, double mean, double stddev)
    : rank_(rank)
    , distribution_(mean, stddev)
  {}

  Dims strides() { return Dims(rank_, 0); }

  void *data() {
    return NULL;
  }

  virtual double get(const std::vector<int64_t> &element) const {
    return distribution_(generator_);
  };

  virtual void set(const std::vector<int64_t> &element, double value) {

  }

  const size_t rank_;

  mutable std::normal_distribution<double> distribution_;
  mutable std::default_random_engine generator_;
};




class GenTensor : public Tensor {
public:
  GenTensor(DataType data_type, const Dims &size,
            const std::optional<std::string> &name,
            double mean, double stddev)
    : Tensor(data_type, size, name)
    , mean_(mean)
    , stddev_(stddev)
  {}

  std::unique_ptr<TensorAccess> access() {
    return std::make_unique<GenTensorAccess>(dims_.size(), mean_, stddev_);
  }

  std::shared_ptr<Tensor> slice(const Dims &offset, const Dims &size) {
    return std::make_shared<GenTensor>(data_type_, size, name_,
                                       mean_, stddev_);
  }

  std::string info() const {
    std::stringstream ss;
    ss << Tensor::info();
    ss << "(mean:" << mean_ << ", stddev:" << stddev_ << ")";
    return ss.str();
  }

  const double mean_;
  const double stddev_;
};





std::shared_ptr<Tensor>
Tensor::find(Tensor::DataType data_type,
             const Dims &size,
             double init_mean,
             double init_stddev,
             Tensors &named_tensors,
             const std::optional<const std::string> &name)
{
  if(name) {
    auto it = named_tensors.find(*name);
    if(it != named_tensors.end()) {
      auto t = it->second;
      assert(t->data_type_ == data_type);
      assert(t->dims_ == size);
      return t;
    }
  }

  return std::make_shared<GenTensor>(data_type, size, name,
                                     init_mean, init_stddev);
}

std::shared_ptr<Tensor>
Tensor::make(Tensor::DataType data_type,
             const Dims &size,
             double init_mean,
             double init_stddev)
{
  return std::make_shared<GenTensor>(data_type, size, std::nullopt,
                                     init_mean, init_stddev);
}






//------------------------------------------------------------------------


class CPUTensorStorage : public TensorStorageAccess {

public:
  CPUTensorStorage(Tensor::DataType data_type, const Dims &size, const Dims &strides)
    : TensorStorageAccess(data_type)
  {
    data_ = calloc(1, size[0] * strides[0] * Tensor::DataTypeSize(data_type));
  }

  ~CPUTensorStorage()
  {
    free(data_);
  }
};



//------------------------------------------------------------------------

class CPUTensorAccess : public TensorAccess {

public:
  CPUTensorAccess(const Dims &strides, std::shared_ptr<CPUTensorStorage> storage,
                  int64_t offset)
    : strides_(strides)
    , storage_(storage)
    , offset_(offset)
  {}

  ~CPUTensorAccess() {}

  Dims strides() { return strides_; }

  void *data() {
    assert(offset_ == 0);
    return storage_->data_;
  }

  size_t offsetForElement(const std::vector<int64_t> &element) const {
    size_t offset = offset_;
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
  int64_t offset_;
};


//------------------------------------------------------------------------


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





class CPUTensor : public Tensor {
public:
  CPUTensor(DataType data_type, const Dims &size,
            const std::optional<std::string> &name)
    : Tensor(data_type, size, name)
    , strides_(computeCPUStrides(size))
    , storage_(std::make_shared<CPUTensorStorage>(data_type, size, strides_))
    , offset_(0)
  {}

  CPUTensor(const Dims &size, const Dims &strides,
            std::shared_ptr<CPUTensorStorage> storage, int64_t offset,
            const std::optional<std::string> &name)
    : Tensor(storage->data_type_, size, name)
    , strides_(strides)
    , storage_(storage)
    , offset_(offset)
  {}

  std::unique_ptr<TensorAccess> access() {
    return std::make_unique<CPUTensorAccess>(strides_, storage_, offset_);
  }

  std::shared_ptr<Tensor> slice(const Dims &offset, const Dims &size);

  virtual std::string info() const;

  const Dims strides_;
  const std::shared_ptr<CPUTensorStorage> storage_;
  const int64_t offset_;
};


std::shared_ptr<Tensor>
CPUTensor::slice(const Dims &offset, const Dims &size)
{
  int64_t o = offset_;

  for(size_t i = 0; i < strides_.size() && i < offset.size(); i++) {
    o += offset[i] * strides_[i];
  }

  return std::make_shared<CPUTensor>(size, strides_,
                                     storage_, o,
                                     namePostfix("slice"));
}


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
makeCPUTensor(Tensor::DataType data_type, const Dims &size,
              const std::optional<const std::string> &name)
{
  return std::make_shared<CPUTensor>(data_type, size, name);
}

}



