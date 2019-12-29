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


#include <sstream>
#include "saga.h"
#include "tensor.h"

#include "cuda_common.h"
#include "cuda_tensor.h"


namespace saga {

class CudaTensorStorage : public TensorStorageAccess {

public:
  CudaTensorStorage(Tensor::DataType data_type, size_t size)
    : TensorStorageAccess(data_type)
    , element_size_(Tensor::DataTypeSize(data_type))
  {
    chkCuda(cudaMallocManaged(&data_, size, cudaMemAttachGlobal));
    chkCuda(cudaMemset(data_, 0, size));
  }

  ~CudaTensorStorage()
  {
    chkCuda(cudaFree(data_));
  }

  void *deviceMem(int64_t offset)
  {
    void *r = (void *)((char *)data_ + offset * element_size_);
    return r;
  }

  const size_t element_size_;
};




class CudaTensorAccess : public TensorAccess {

public:
  CudaTensorAccess(std::shared_ptr<CudaTensorStorage> storage,
                   cudnnTensorDescriptor_t desc,
                   int64_t offset)
    : storage_(storage)
    , offset_(offset)
  {
    const int max_rank = 8;
    int dims[max_rank];
    int strides[max_rank];
    int rank;
    cudnnDataType_t data_type;

    chkCUDNN(cudnnGetTensorNdDescriptor(desc, max_rank, &data_type,
                                        &rank, dims, strides));

    for(int i = 0; i < rank; i++) {
      strides_.push_back(strides[i]);
    }

    cudaDeviceSynchronize();
  }

  ~CudaTensorAccess() {}

  Dims strides() { return strides_; }

  void *data() { return storage_->data_; }

  int64_t offsetForElement(const std::vector<int64_t> &element) const {
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

  Dims strides_;
  const std::shared_ptr<CudaTensorStorage> storage_;
  const int64_t offset_;
};



cudnnDataType_t
cudnnDataType_from_dataType(Tensor::DataType data_type)
{
  switch(data_type) {
  case Tensor::DataType::FLOAT:
    return CUDNN_DATA_FLOAT;
  case Tensor::DataType::HALF:
    return CUDNN_DATA_HALF;
  case Tensor::DataType::U8:
    return CUDNN_DATA_UINT8;
  default:
    fprintf(stderr, "Unsupported data_type %d for cuda tensor\n",
            (int)data_type);
    abort();
  }
}


CudaTensor::CudaTensor(const std::string &name, DataType data_type, Dims dims,
                       cudnnTensorFormat_t format)
  : Tensor(name, data_type, dims)
  , type_(cudnnDataType_from_dataType(data_type))
  , offset_(0)
{
  chkCUDNN(cudnnCreateTensorDescriptor(&desc_));
  assert(dims.size() >= 0 && dims.size() <= 4);
  chkCUDNN(cudnnSetTensor4dDescriptor(desc_, format, type_,
                                      dims[0],
                                      dims.size() > 1 ? dims[1] : 1,
                                      dims.size() > 2 ? dims[2] : 1,
                                      dims.size() > 3 ? dims[3] : 1));

  size_t bytes;
  chkCUDNN(cudnnGetTensorSizeInBytes(desc_, &bytes));

  storage_ = std::make_shared<CudaTensorStorage>(data_type, bytes);
}


CudaTensor::CudaTensor(const std::string &name,
                       std::shared_ptr<CudaTensorStorage> storage,
                       Dims dims, cudnnTensorFormat_t format)
  : Tensor(name, storage->data_type_, dims)
  , type_(cudnnDataType_from_dataType(storage->data_type_))
  , offset_(0)
{
  chkCUDNN(cudnnCreateTensorDescriptor(&desc_));
  assert(dims.size() >= 0 && dims.size() <= 4);
  chkCUDNN(cudnnSetTensor4dDescriptor(desc_, format, type_,
                                      dims[0],
                                      dims.size() > 1 ? dims[1] : 1,
                                      dims.size() > 2 ? dims[2] : 1,
                                      dims.size() > 3 ? dims[3] : 1));
  storage_ = storage;
}


CudaTensor::CudaTensor(const std::string &name,
                       std::shared_ptr<CudaTensor> alias,
                       Dims dims, std::vector<int64_t> offset_element)
  : Tensor(name, alias->storage_->data_type_, dims)
  , type_(cudnnDataType_from_dataType(alias->storage_->data_type_))
  , offset_(alias->offset_)
  , storage_(alias->storage_)
{
  const int max_rank = 8;
  int dimsA[max_rank];
  int stridesA[max_rank];
  int rank;
  cudnnDataType_t data_type;

  chkCUDNN(cudnnGetTensorNdDescriptor(alias->desc_, max_rank, &data_type,
                                      &rank, dimsA, stridesA));

  assert(data_type == type_);
  assert((size_t)rank == dims.size());

  for(int i = 0; i < rank; i++) {
    dimsA[i] = dims[i];
  }
  chkCUDNN(cudnnCreateTensorDescriptor(&desc_));
  chkCUDNN(cudnnSetTensorNdDescriptor(desc_, type_, rank,
                                      dimsA, stridesA));

  for(int i = 0; i < rank && i < (ssize_t)offset_element.size(); i++) {
    offset_ += offset_element[i] * stridesA[i];
  }
}

CudaTensor::~CudaTensor()
{
  chkCUDNN(cudnnDestroyTensorDescriptor(desc_));
}

std::unique_ptr<TensorAccess>
CudaTensor::access()
{
  return std::make_unique<CudaTensorAccess>(storage_, desc_, offset_);
}


void *
CudaTensor::deviceMem() const
{
  return storage_->deviceMem(offset_);
};


std::string
CudaTensor::info() const
{
  std::stringstream ss;
  ss << "\"" << name_ << "\"";

  const int max_rank = 8;
  int dims[max_rank];
  int strides[max_rank];
  int rank;
  cudnnDataType_t data_type;

  chkCUDNN(cudnnGetTensorNdDescriptor(desc_, max_rank, &data_type,
                                      &rank, dims, strides));
  switch(data_type) {
  case CUDNN_DATA_FLOAT:
    ss << "<float>";
    break;
  case CUDNN_DATA_HALF:
    ss << "<half>";
    break;
  case CUDNN_DATA_UINT8:
    ss << "<u8>";
    break;
  default:
    ss << "<?>";
    break;
  }

  const char *prefix = "";
  ss << "[";
  for(int i = 0; i < rank; i++) {
    ss << prefix << dims[i];
    prefix = ", ";
  }
  ss << "]";

  prefix = "";
  ss << "{";
  for(int i = 0; i < rank; i++) {
    ss << prefix << strides[i];
    prefix = ", ";
  }
  ss << "}@cuda:" << storage_->data_;
  return ss.str();
}

}
