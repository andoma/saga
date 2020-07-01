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

#include "context.h"
#include "cuda_common.h"
#include "cuda_tensor.h"


namespace saga {



CudaTensorStorage::CudaTensorStorage(Tensor::DataType data_type, size_t size,
                                     const std::shared_ptr<CudaContext> &ctx)
  : TensorStorage(data_type)
  , ctx_(ctx)
  , size_(size)
  , element_size_(Tensor::DataTypeSize(data_type))
  , id_(++ctx->tensor_storage_id_gen_)
{
}


CudaTensorStorage::~CudaTensorStorage()
{
  if(mem_)
    assert(mem_ == data_);
  chkCuda(cudaFree(mem_));
}

void
CudaTensorStorage::alloc()
{
  if(mem_)
    return;
  chkCuda(cudaMallocManaged(&mem_, size_, cudaMemAttachGlobal));
  chkCuda(cudaMemset(mem_, 0, size_));
  data_ = mem_;
}

void
CudaTensorStorage::setTmpMem(void *p)
{
  if(mem_)
    return;
  data_ = p;
}

void *CudaTensorStorage::deviceMem(int64_t offset)
{
  return (void *)((char *)data_ + offset * element_size_);
}



class CudaTensorAccess : public TensorAccess {

public:
  CudaTensorAccess(std::shared_ptr<CudaTensorStorage> storage,
                   cudnnTensorDescriptor_t desc,
                   int64_t offset)
    : storage_(storage)
    , offset_(offset)
    , sync_(false)
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

    storage_->ctx_->mutex_.lock();
  }

  ~CudaTensorAccess() {
    storage_->ctx_->mutex_.unlock();
  }

  Dims strides() { return strides_; }

  void *data() { return storage_->data(); }

  int64_t offsetForElement(const Dims &element) const {
    size_t offset = offset_;
    for(size_t i = 0; i < element.size() && i < strides_.size(); i++) {
      offset += element[i] * strides_[i];
    }
    return offset;
  }

  virtual double get(const Dims &element) {
    if(!sync_) {
      cudaStreamSynchronize(storage_->ctx_->stream_);
      sync_ = true;
    }
    return storage_->get(offsetForElement(element));
  };

  virtual void set(const Dims &element, double value) {
    storage_->set(offsetForElement(element), value);
  }

  virtual void copyBytesFrom(const Dims &element,
                             const void *data, size_t size) {
    const size_t o = offsetForElement(element) * storage_->element_size_;
    char *dst = (char *)storage_->data();
    cudaMemcpy(dst + o, data, size, cudaMemcpyHostToDevice);
  }

  Dims strides_;
  const std::shared_ptr<CudaTensorStorage> storage_;
  const int64_t offset_;
  bool sync_;
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
  case Tensor::DataType::I32:
    return CUDNN_DATA_INT32;
  default:
    fprintf(stderr, "Unsupported data_type %d for cuda tensor\n",
            (int)data_type);
    abort();
  }
}


CudaTensor::CudaTensor(DataType data_type, const Dims &size,
                       cudnnTensorFormat_t format,
                       const std::shared_ptr<CudaContext> &ctx,
                       const std::optional<const std::string> &name)
  : Tensor(data_type, size, name)
  , type_(cudnnDataType_from_dataType(data_type))
  , offset_(0)
{
  chkCUDNN(cudnnCreateTensorDescriptor(&desc_));
  assert(size.size() >= 0 && size.size() <= 4);
  chkCUDNN(cudnnSetTensor4dDescriptor(desc_, format, type_,
                                      size[0],
                                      size.size() > 1 ? size[1] : 1,
                                      size.size() > 2 ? size[2] : 1,
                                      size.size() > 3 ? size[3] : 1));

  size_t bytes;
  chkCUDNN(cudnnGetTensorSizeInBytes(desc_, &bytes));

  storage_ = std::make_shared<CudaTensorStorage>(data_type, bytes, ctx);
}


CudaTensor::CudaTensor(std::shared_ptr<CudaTensorStorage> storage,
                       const Dims &size, cudnnTensorFormat_t format,
                       const std::optional<const std::string> &name)
  : Tensor(storage->data_type_, size, name)
  , type_(cudnnDataType_from_dataType(storage->data_type_))
  , offset_(0)
{
  chkCUDNN(cudnnCreateTensorDescriptor(&desc_));
  assert(size.size() >= 0 && size.size() <= 4);
  chkCUDNN(cudnnSetTensor4dDescriptor(desc_, format, type_,
                                      size[0],
                                      size.size() > 1 ? size[1] : 1,
                                      size.size() > 2 ? size[2] : 1,
                                      size.size() > 3 ? size[3] : 1));
  storage_ = storage;
}


CudaTensor::CudaTensor(std::shared_ptr<CudaTensor> alias,
                       const Dims &size, std::vector<int64_t> offset_element,
                       const std::optional<const std::string> &name)
  : Tensor(alias->storage_->data_type_, size, name)
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
  for(size_t i = 0; i < size.size(); i++) {
    dimsA[i] = size[i];
  }

  chkCUDNN(cudnnCreateTensorDescriptor(&desc_));
  chkCUDNN(cudnnSetTensorNdDescriptor(desc_, type_,rank, dimsA, stridesA));

  for(size_t i = 0; i < size.size() && i < offset_element.size(); i++) {
    offset_ += offset_element[i] * stridesA[i];
  }
}

CudaTensor::CudaTensor(std::shared_ptr<CudaTensorStorage> storage,
                       const Dims &size,
                       int64_t offset,
                       const int *strides,
                       const std::optional<const std::string> &name)
  : Tensor(storage->data_type_, size, name)
  , type_(cudnnDataType_from_dataType(storage->data_type_))
  , offset_(offset)
  , storage_(storage)
{
  chkCUDNN(cudnnCreateTensorDescriptor(&desc_));
  chkCUDNN(cudnnSetTensorNdDescriptor(desc_, type_, size.size(),
                                      &size[0], strides));
}


CudaTensor::CudaTensor(DataType data_type,
                       const Dims &size,
                       const int *strides,
                       const std::shared_ptr<CudaContext> &ctx,
                       const std::optional<const std::string> &name)
  : Tensor(data_type, size, name)
  , type_(cudnnDataType_from_dataType(data_type))
{
  chkCUDNN(cudnnCreateTensorDescriptor(&desc_));
  chkCUDNN(cudnnSetTensorNdDescriptor(desc_, type_, size.size(),
                                      &size[0], strides));
  size_t bytes;
  chkCUDNN(cudnnGetTensorSizeInBytes(desc_, &bytes));

  storage_ = std::make_shared<CudaTensorStorage>(data_type, bytes, ctx);
}


CudaTensor::CudaTensor(const CudaTensor &o,
                       cudnnTensorFormat_t format,
                       const std::optional<const std::string> &postfix)
  : CudaTensor(o.data_type_, o.dims_, format, o.storage_->ctx_,
               postfix ? o.namePostfix(*postfix) : std::nullopt)
{
}

CudaTensor::CudaTensor(const CudaTensor &o,
                       const std::optional<const std::string> &name)
  : Tensor(o.data_type_, o.dims_, name)
  , type_(cudnnDataType_from_dataType(data_type_))
  , offset_(0)
{
  const int max_rank = 8;
  int dimsA[max_rank];
  int stridesA[max_rank];
  int rank;
  cudnnDataType_t data_type;

  chkCUDNN(cudnnGetTensorNdDescriptor(o.desc_, max_rank, &data_type,
                                      &rank, dimsA, stridesA));

  chkCUDNN(cudnnCreateTensorDescriptor(&desc_));
  chkCUDNN(cudnnSetTensorNdDescriptor(desc_, type_, rank,
                                      dimsA, stridesA));
  size_t bytes;
  chkCUDNN(cudnnGetTensorSizeInBytes(desc_, &bytes));

  storage_ = std::make_shared<CudaTensorStorage>(data_type_, bytes,
                                                 o.storage_->ctx_);
}


CudaTensor::CudaTensor(DataType data_type,
                       const CudaTensor &o,
                       const std::optional<const std::string> &name)
  : Tensor(data_type, o.dims_, name)
  , type_(cudnnDataType_from_dataType(data_type))
  , offset_(0)
{
  const int max_rank = 8;
  int dimsA[max_rank];
  int stridesA[max_rank];
  int rank;
  cudnnDataType_t data_type_o;

  chkCUDNN(cudnnGetTensorNdDescriptor(o.desc_, max_rank, &data_type_o,
                                      &rank, dimsA, stridesA));

  chkCUDNN(cudnnCreateTensorDescriptor(&desc_));
  chkCUDNN(cudnnSetTensorNdDescriptor(desc_, type_, rank,
                                      dimsA, stridesA));
  size_t bytes;
  chkCUDNN(cudnnGetTensorSizeInBytes(desc_, &bytes));

  storage_ = std::make_shared<CudaTensorStorage>(data_type_, bytes,
                                                 o.storage_->ctx_);
}


CudaTensor::~CudaTensor()
{
  chkCUDNN(cudnnDestroyTensorDescriptor(desc_));
}

std::unique_ptr<TensorAccess>
CudaTensor::access()
{
  if(!storage_->data())
    return nullptr;
  return std::make_unique<CudaTensorAccess>(storage_, desc_, offset_);
}

std::shared_ptr<Tensor>
CudaTensor::slice(const Dims &offset, const Dims &size)
{
  const int max_rank = 8;
  int dimsA[max_rank];
  int stridesA[max_rank];
  int rank;
  cudnnDataType_t data_type;

  chkCUDNN(cudnnGetTensorNdDescriptor(desc_, max_rank, &data_type,
                                      &rank, dimsA, stridesA));

  int64_t o = offset_;

  for(int i = 0; i < rank && i < (ssize_t)offset.size(); i++) {
    o += offset[i] * stridesA[i];
  }

  return std::make_shared<CudaTensor>(storage_, size, o, stridesA,
                                      namePostfix("slice"));
}



std::shared_ptr<CudaTensor>
CudaTensor::makeSharedGrad()
{
  if(!grad_)
    grad_ = std::make_shared<CudaTensor>(*this, namePostfix("shared_grad"));

  return grad_;
}


std::shared_ptr<CudaTensor>
CudaTensor::makePrivateGrad()
{
  return std::make_shared<CudaTensor>(*this, namePostfix("private_grad"));
}


void *
CudaTensor::deviceMem() const
{
  return storage_->deviceMem(offset_);
};


std::string
CudaTensor::shortname() const
{
  std::stringstream ss;
  ss << "T" << storage_->id_;
  if(name_) {
    ss << "\"" << *name_ << "\"";
  }

  if(offset_) {
    ss << "[" << offset_ << "]";
  }

  return ss.str();
}


std::string
CudaTensor::info() const
{
  std::stringstream ss;
  ss << "T" << storage_->id_;
  if(name_)
    ss << "\"" << *name_ << "\"";

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
  case CUDNN_DATA_INT32:
    ss << "<i32>";
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
  ss << "}@cuda:" << storage_->deviceMem(0);

  if(storage_->mem_) {
    ss << "<static>";
  }

  if(offset_) {
    ss << " + " << offset_;
  }
  return ss.str();
}

bool CudaTensor::cpacked() const
{
  const int max_rank = 8;
  int dims[max_rank];
  int strides[max_rank];
  int rank;
  cudnnDataType_t data_type;

  chkCUDNN(cudnnGetTensorNdDescriptor(desc_, max_rank, &data_type,
                                      &rank, dims, strides));

  return strides[1] == 1;
}

void
CudaTensor::copyFromLocked(Tensor &t, int dst_broadcast_dimension)
{
  const int max_rank = 8;
  int dims[max_rank];
  int strides[max_rank];
  int rank;
  cudnnDataType_t data_type;

  auto ta = t.access();
  if(ta == nullptr) {
    return;
  }

  storage_->alloc();

  chkCUDNN(cudnnGetTensorNdDescriptor(desc_, max_rank, &data_type,
                                      &rank, dims, strides));

  cudaStreamSynchronize(storage_->ctx_->stream_);

  if(!copy_tensor(storage_->deviceMem(offset_),
                  dims_.size(),
                  &dims_[0],
                  &strides[0],
                  data_type_,
                  t, ta.get(),
                  dst_broadcast_dimension)) {
    fprintf(stderr,
            "Cuda Tensor copy failed\n"
            "From: %s\n"
            "  To: %s\n",
            t.info().c_str(),
            info().c_str());
    abort();
  }
}


size_t
size_from_params(const Dims &size, cudnnTensorFormat_t format,
                 cudnnDataType_t data_type)
{
  cudnnTensorDescriptor_t desc;
  chkCUDNN(cudnnCreateTensorDescriptor(&desc));
  assert(size.size() >= 0 && size.size() <= 4);

  chkCUDNN(cudnnSetTensor4dDescriptor(desc, format, data_type,
                                      size[0],
                                      size.size() > 1 ? size[1] : 1,
                                      size.size() > 2 ? size[2] : 1,
                                      size.size() > 3 ? size[3] : 1));

  size_t bytes;
  chkCUDNN(cudnnGetTensorSizeInBytes(desc, &bytes));

  cudnnDestroyTensorDescriptor(desc);
  return bytes;
}


CudaTensorStorageDoubleBuffered::CudaTensorStorageDoubleBuffered(Tensor::DataType data_type,
                                                                 Dims &dims,
                                                                 cudnnTensorFormat_t format,
                                                                 const std::shared_ptr<CudaContext> &ctx)
  : CudaTensorStorage(data_type, size_from_params(dims, format, cudnnDataType_from_dataType(data_type)), ctx)
  , index_(0)
{

  for(int i = 0; i < 2; i++) {
    chkCuda(cudaMallocManaged(&buffers_[i], size_, cudaMemAttachGlobal));
    chkCuda(cudaMemset(buffers_[i], 0, size_));
  }
}


CudaTensorStorageDoubleBuffered::~CudaTensorStorageDoubleBuffered()
{
  for(int i = 0; i < 2; i++) {
    chkCuda(cudaFree(buffers_[i]));
  }
}


void *
CudaTensorStorageDoubleBuffered::deviceMem(int64_t offset)
{
  void *buf = buffers_[index_ & 1];

  void *r = (void *)((char *)buf + offset * element_size_);
  return r;
}


void *
CudaTensorStorageDoubleBuffered::data(int buffer) const
{
  return buffers_[(buffer + index_) & 1];
}


double
CudaTensorStorageDoubleBuffered::get(size_t offset) const {
  return get_(data(0), offset);
}

double
CudaTensorStorageDoubleBuffered::get(size_t offset, int buffer) const {
  return get_(data(buffer), offset);
}


void
CudaTensorStorageDoubleBuffered::set(size_t offset, double value)
{
  set_(data(0), offset, value);
}

void
CudaTensorStorageDoubleBuffered::set(size_t offset, double value, int buffer)
{
  set_(data(buffer), offset, value);
}

int
CudaTensorStorageDoubleBuffered::flip()
{
  index_++;
  return index_ & 1;
}

void
CudaTensorStorageDoubleBuffered::prefetchGPU()
{
  cudaMemPrefetchAsync(data(1), size_, ctx_->deviceId_, ctx_->stream_);
}






class CudaTensorBatchAccess : public TensorAccess {

  static const int MAX_RANK = 8;

public:
  CudaTensorBatchAccess(CudaTensorStorageDoubleBuffered *storage,
                        cudnnTensorDescriptor_t desc,
                        int64_t offset)
    : storage_(storage)
    , offset_(offset)
  {
    int dims[MAX_RANK];
    cudnnDataType_t data_type;

    chkCUDNN(cudnnGetTensorNdDescriptor(desc, MAX_RANK, &data_type,
                                        &rank_, dims, strides_));
  }

  int64_t offsetForElement(const Dims &element) const {
    size_t offset = offset_;
    for(int i = 0; i < (int)element.size() && i < rank_; i++) {
      offset += element[i] * strides_[i];
    }
    return offset;
  }

  Dims strides() override {
    abort();
  }

  void *data() override {
    abort();
  };

  void copyBytesFrom(const Dims &element,
                     const void *data, size_t size) override {
    const size_t o = offsetForElement(element) * storage_->element_size_;
    char *dst = (char *)storage_->data(1);
    cudaMemcpy(dst + o, data, size, cudaMemcpyHostToDevice);
  }

  void *getAddr(const Dims &element) override {
    size_t off = offsetForElement(element) * storage_->element_size_;
    return (void *)((char *)storage_->data(1) + off);
  };

  double get(const Dims &element) override {
    return storage_->get(offsetForElement(element), 1);
  };

  void set(const Dims &element, double value) override {
    storage_->set(offsetForElement(element), value, 1);
  }


  CudaTensorStorageDoubleBuffered *storage_;
  int64_t offset_;
  int rank_;
  int strides_[MAX_RANK];
};


void
CudaProgram::issueOps(const CudaBatchAccessOps ops, long batch)
{
  for(const auto &op : ops) {
    CudaTensorBatchAccess ta(op.storage_.get(),
                             op.tensor_->desc_,
                             op.tensor_->offset_);
    op.fn_(ta, batch);
#if 0
    if(op.prefetch_)
      op.storage_->prefetchGPU();
#endif
  }
}

void
CudaProgram::flipDoubleBufferedTensors()
{
  for(const auto &s : flips_) {
    s->flip();
  }
}


}
