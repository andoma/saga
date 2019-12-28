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
#include <cudnn.h>
#include <cublas_v2.h>

#include "saga.h"
#include "tensor.h"




#define chkCUDNN(expression) {                                          \
    const cudnnStatus_t cudnn_status__ = (expression);                  \
    if(cudnn_status__ != CUDNN_STATUS_SUCCESS) {                        \
      fprintf(stderr, "CUDNN error at %s:%d in %s: %s\n",               \
              __FILE__, __LINE__, __FUNCTION__,                         \
              cudnnGetErrorString(cudnn_status__));                     \
      abort();                                                          \
    }                                                                   \
  }


#define chkCuda(expression) {                                           \
    int cuda_status__ = (expression);                                   \
    if(cuda_status__) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d in %s\n",                    \
              __FILE__, __LINE__, __FUNCTION__);                        \
    }                                                                   \
  }




namespace saga {

class CudnnContext {
public:
  void init();

  cudnnHandle_t cudnn_;
  cublasHandle_t cublas_;


  void *workspace_;
  size_t workspace_size_;
};



void
CudnnContext::init()
{
  int device;
  cudaGetDevice(&device);

  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device);

  printf("Device name: %s\n", prop.name);
  printf("Compute capability: %d.%d\n", prop.major, prop.minor);
  printf("CanMapHostMem: %d\n", prop.canMapHostMemory);
  printf("ComputeMode: %d\n", prop.computeMode);
  printf("Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
  printf("Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
  printf("Peak Memory Bandwidth (GB/s): %f\n\n",
         2.0 * prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);

  chkCUDNN(cudnnCreate(&cudnn_));
  chkCuda(cublasCreate(&cublas_));

  chkCuda(cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH));
}



class CudaTensorStorage : public TensorStorageAccess {

public:
  CudaTensorStorage(Tensor::DataType data_type, size_t size)
    : TensorStorageAccess(data_type)
  {
    chkCuda(cudaMallocManaged(&data_, size, cudaMemAttachGlobal));
    chkCuda(cudaMemset(data_, 0, size));
  }

  ~CudaTensorStorage()
  {
    chkCuda(cudaFree(data_));
  }
};




class CudaTensorAccess : public TensorAccess {

public:
  CudaTensorAccess(std::shared_ptr<CudaTensorStorage> storage,
                   cudnnTensorDescriptor_t desc)
    : storage_(storage)
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

  Dims strides_;
  const std::shared_ptr<CudaTensorStorage> storage_;
};




static Dims
computeCudaStrides(const Dims &dims)
{
  Dims strides;
  int stride = 1;
  for(int i = dims.size() - 1; i >= 0; i--) {
    strides.insert(strides.begin(), stride);
    stride *= dims[i];
  }
  return strides;
}


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


class CudaTensor : public Tensor {

public:
  CudaTensor(const std::string &name, DataType data_type, Dims dims,
             cudnnTensorFormat_t format)
    : Tensor(name, data_type, dims)
    , type_(cudnnDataType_from_dataType(data_type))
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


  CudaTensor(const std::string &name,
             std::shared_ptr<CudaTensorStorage> storage,
             Dims dims, cudnnTensorFormat_t format)
    : Tensor(name, storage->data_type_, dims)
    , type_(cudnnDataType_from_dataType(storage->data_type_))
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

  std::unique_ptr<TensorAccess> access() {
    return std::make_unique<CudaTensorAccess>(storage_, desc_);
  }

  cudnnTensorDescriptor_t desc() const {
    return desc_;
  }

  void *deviceMem() const {
    return storage_->data_;
  };

  virtual std::string info() const;

  const cudnnDataType_t type_;
  cudnnTensorDescriptor_t desc_;
  std::shared_ptr<CudaTensorStorage> storage_;
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



//------------------------------------------------------------------------

class CudnnProgram : public Program {
public:

  CudnnProgram()
    : workspace_size_(0)
  {}

  void exec();

  std::unordered_map<std::shared_ptr<Tensor>,
                     std::shared_ptr<CudaTensor>> tensors_;

  std::shared_ptr<CudnnContext> ctx_;
  size_t workspace_size_;

};



static std::shared_ptr<CudaTensor>
lower_tensor(CudnnProgram &p, std::shared_ptr<Tensor> src,
             size_t leading_dimensions = 0)
{
  if(src == nullptr)
    return nullptr;

  auto it = p.tensors_.find(src);
  if(it != p.tensors_.end()) {
    printf("Tensor %s %p already lowered to %p\n",
           src->name_.c_str(), src.get(), it->second.get());
    return it->second;
  }

  std::vector<int64_t> dims(leading_dimensions, 1);
  dims.insert(dims.end(),  src->dims_.begin(),  src->dims_.end());

  auto t = std::make_shared<CudaTensor>(src->name_, src->data_type_,
                                        dims, CUDNN_TENSOR_NHWC);

  t->copyFrom(*src);
  p.tensors_[src] = t;
  printf("Tensor %s (%s) %p lowered to %p\n",
         src->name_.c_str(), src->info().c_str(),src.get(), t.get());
  return t;
}


//------------------------------------------------------------------------

struct CudnnConvolutionFwd : public Operation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, w_, b_, y_;

  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionFwdAlgo_t conv_fwd_algo_;

  ~CudnnConvolutionFwd()
  {
    chkCUDNN(cudnnDestroyFilterDescriptor(filter_desc_));
    chkCUDNN(cudnnDestroyConvolutionDescriptor(conv_desc_));
  }

  CudnnConvolutionFwd(CudnnProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(lower_tensor(p, n.inputs_.get("x")))
    , w_(lower_tensor(p, n.inputs_.get("w")))
    , b_(lower_tensor(p, n.inputs_.get("b")))
    , y_(lower_tensor(p, n.outputs_.get("y")))
  {
    chkCUDNN(cudnnCreateFilterDescriptor(&filter_desc_));
    chkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));


    chkCUDNN(cudnnSetFilter4dDescriptor(filter_desc_,
                                        x_->type_,
                                        CUDNN_TENSOR_NHWC,
                                        w_->dims_[0],
                                        w_->dims_[1],
                                        w_->dims_[2],
                                        w_->dims_[3]));

    chkCUDNN(cudnnSetConvolutionMathType(conv_desc_,
                                         CUDNN_TENSOR_OP_MATH));

    const int pad = n.attributes_.get("pad", 0);
    const int stride = n.attributes_.get("stride", 1);

    chkCUDNN(cudnnSetConvolution2dDescriptor(conv_desc_,
                                             pad, pad,
                                             stride, stride,
                                             1, 1,
                                             CUDNN_CROSS_CORRELATION,
                                             CUDNN_DATA_FLOAT));

    chkCUDNN(cudnnGetConvolutionForwardAlgorithm(ctx_->cudnn_,
                                                 x_->desc_,
                                                 filter_desc_,
                                                 conv_desc_,
                                                 y_->desc_,
                                                 CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                 0,
                                                 &conv_fwd_algo_));

    size_t workspace_bytes;
    chkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(ctx_->cudnn_,
                                                     x_->desc_,
                                                     filter_desc_,
                                                     conv_desc_,
                                                     y_->desc_,
                                                     conv_fwd_algo_,
                                                     &workspace_bytes));

    p.workspace_size_ = std::max(p.workspace_size_, workspace_bytes);
  }

  void exec() {
    float alpha = 1.0f, beta = 0.0f;


    chkCUDNN(cudnnConvolutionForward(ctx_->cudnn_, &alpha,
                                     x_->desc(),
                                     x_->deviceMem(),
                                     filter_desc_,
                                     w_->deviceMem(),
                                     conv_desc_,
                                     conv_fwd_algo_,
                                     ctx_->workspace_, ctx_->workspace_size_,
                                     &beta,
                                     y_->desc(),
                                     y_->deviceMem()));

    if(b_) {
      chkCUDNN(cudnnAddTensor(ctx_->cudnn_,
                              &alpha, b_->desc(), b_->deviceMem(),
                              &alpha, y_->desc(), y_->deviceMem()));
    }


    printf("conv x: %s\n", x_->statsString().c_str());
    printf("conv w: %s\n", w_->statsString().c_str());
    if(b_)
      printf("conv b: %s\n", b_->statsString().c_str());
    printf("conv y: %s\n", y_->statsString().c_str());


  }

};


//------------------------------------------------------------------------

struct CudnnBatchNormFwd : public Operation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, s_, b_, m_, v_, y_;
  const float epsilon_;

  CudnnBatchNormFwd(CudnnProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(lower_tensor(p, n.inputs_.get("x")))
    , s_(lower_tensor(p, n.inputs_.get("s"), 1))
    , b_(lower_tensor(p, n.inputs_.get("b"), 1))
    , m_(lower_tensor(p, n.inputs_.get("m"), 1))
    , v_(lower_tensor(p, n.inputs_.get("v"), 1))
    , y_(lower_tensor(p, n.outputs_.get("y")))
    , epsilon_(n.attributes_.get("epsilon", 1e-5f))
  {
  }

  void exec() {
    float alpha = 1.0f, beta = 0.0f;
#if 0
    x_->print("x");
    s_->print("s");
    b_->print("b");
    rm_->print("rm");
    riv_->print("riv");
    y_->print("y");
    printf("x: %s\n", x_->info().c_str());
    printf("s: %s\n", s_->info().c_str());
    printf("b: %s\n", b_->info().c_str());
    printf("m: %s\n", m_->info().c_str());
    printf("v: %s\n", v_->info().c_str());
    printf("y: %s\n", y_->info().c_str());
#endif

    chkCUDNN(cudnnBatchNormalizationForwardInference(ctx_->cudnn_,
                                                     CUDNN_BATCHNORM_SPATIAL,
                                                     &alpha, &beta,
                                                     x_->desc(),
                                                     x_->deviceMem(),
                                                     y_->desc(),
                                                     y_->deviceMem(),
                                                     s_->desc(),
                                                     s_->deviceMem(),
                                                     b_->deviceMem(),
                                                     m_->deviceMem(),
                                                     v_->deviceMem(),
                                                     epsilon_));

  }
};

//------------------------------------------------------------------------

struct CudnnReluFwd : public Operation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;
  cudnnActivationDescriptor_t desc_;

  CudnnReluFwd(CudnnProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(lower_tensor(p, n.inputs_.get("x")))
    , y_(lower_tensor(p, n.outputs_.get("y")))
  {
    chkCUDNN(cudnnCreateActivationDescriptor(&desc_));
    chkCUDNN(cudnnSetActivationDescriptor(desc_, CUDNN_ACTIVATION_RELU,
                                          CUDNN_PROPAGATE_NAN, 0.0));
  }

  ~CudnnReluFwd()
  {
    chkCUDNN(cudnnDestroyActivationDescriptor(desc_));
  }

  void exec() {
    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnActivationForward(ctx_->cudnn_, desc_,
                                    &alpha,
                                    x_->desc(), x_->deviceMem(),
                                    &beta,
                                    y_->desc(), y_->deviceMem()));
  }
};


//------------------------------------------------------------------------

struct CudnnPoolingFwd : public Operation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;
  cudnnPoolingDescriptor_t desc_;

  CudnnPoolingFwd(CudnnProgram &p, const Node &n, cudnnPoolingMode_t mode)
    : ctx_(p.ctx_)
    , x_(lower_tensor(p, n.inputs_.get("x")))
    , y_(lower_tensor(p, n.outputs_.get("y")))
  {
    const int size   = n.attributes_.get("size", 1);
    const int pad    = n.attributes_.get("pad", 0);
    const int stride = n.attributes_.get("stride", 1);

    chkCUDNN(cudnnCreatePoolingDescriptor(&desc_));

    chkCUDNN(cudnnSetPooling2dDescriptor(desc_,
                                         mode,
                                         CUDNN_PROPAGATE_NAN,
                                         size, size,
                                         pad, pad,
                                         stride, stride));
  }

  ~CudnnPoolingFwd()
  {
    chkCUDNN(cudnnDestroyPoolingDescriptor(desc_));
  }

  void exec() {
    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnPoolingForward(ctx_->cudnn_, desc_,
                                 &alpha,
                                 x_->desc(), x_->deviceMem(),
                                 &beta,
                                 y_->desc(), y_->deviceMem()));
  }
};


//------------------------------------------------------------------------

struct CudnnSumFwd : public Operation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x0_, x1_, y_;
  cudnnOpTensorDescriptor_t desc_;

  CudnnSumFwd(CudnnProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x0_(lower_tensor(p, n.inputs_.get("x0")))
    , x1_(lower_tensor(p, n.inputs_.get("x1")))
    , y_(lower_tensor(p, n.outputs_.get("y")))
  {
    cudnnCreateOpTensorDescriptor(&desc_);
    cudnnSetOpTensorDescriptor(desc_,
                               CUDNN_OP_TENSOR_ADD,
                               CUDNN_DATA_FLOAT,
                               CUDNN_PROPAGATE_NAN);
  }

  ~CudnnSumFwd()
  {
    cudnnDestroyOpTensorDescriptor(desc_);
  }

  void exec() {

    float alpha = 1.0f;
    float beta = 0.0f;

    cudnnOpTensor(ctx_->cudnn_, desc_,
                  &alpha,
                  x0_->desc(),
                  x0_->deviceMem(),
                  &alpha,
                  x1_->desc(),
                  x1_->deviceMem(),
                  &beta,
                  y_->desc(),
                  y_->deviceMem());
  }
};

//------------------------------------------------------------------------

struct CudnnFcFwd : public Operation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, w_, b_, y_;
  const int n_;
  const int num_inputs_;
  const int num_outputs_;
  cudnnOpTensorDescriptor_t desc_;

  CudnnFcFwd(CudnnProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(lower_tensor(p, n.inputs_.get("x")))
    , w_(lower_tensor(p, n.inputs_.get("w")))
    , b_(lower_tensor(p, n.inputs_.get("b"), 1))
    , y_(lower_tensor(p, n.outputs_.get("y")))
    , n_(x_->dims_[0])
    , num_inputs_(x_->dims_[1])
    , num_outputs_(y_->dims_[1])
  {
  }

  void exec() {

    float alpha = 1.0f, beta = 0.0f;
    __half halpha = 1.0f, hbeta = 0.0f;

    switch(x_->type_) {
    case CUDNN_DATA_FLOAT:
      chkCuda(cublasSgemm(ctx_->cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                          num_outputs_, n_, num_inputs_,
                          &alpha,
                          (const float *)w_->deviceMem(), num_inputs_,
                          (const float *)x_->deviceMem(), num_inputs_,
                          &beta,
                          (float *)y_->deviceMem(), num_outputs_));
      break;
    case CUDNN_DATA_HALF:
      chkCuda(cublasHgemm(ctx_->cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                          num_outputs_, n_, num_inputs_,
                          &halpha,
                          (const __half *)w_->deviceMem(), num_inputs_,
                          (const __half *)x_->deviceMem(), num_inputs_,
                          &hbeta,
                          (__half *)y_->deviceMem(), num_outputs_));
      break;
    default:
      abort();
    }

    chkCUDNN(cudnnAddTensor(ctx_->cudnn_,
                            &alpha, b_->desc(), b_->deviceMem(),
                            &alpha, y_->desc(), y_->deviceMem()));
  }
};


//------------------------------------------------------------------------

struct CudnnSoftmaxFwd : public Operation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;

  CudnnSoftmaxFwd(CudnnProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(lower_tensor(p, n.inputs_.get("x")))
    , y_(lower_tensor(p, n.outputs_.get("y")))
  {
  }

  void exec() {

    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnSoftmaxForward(ctx_->cudnn_,
                                 CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha,
                                 x_->desc(), x_->deviceMem(),
                                 &beta,
                                 y_->desc(), y_->deviceMem()));

    printf("softmax x: %s\n", x_->statsString().c_str());
    printf("softmax y: %s %s\n", y_->statsString().c_str(),
           y_->info().c_str());
  }
};




//------------------------------------------------------------------------


static void
generate_forward_operation(CudnnProgram &p, const Node &n)
{
  std::shared_ptr<Operation> o;

  if(n.type_ == "conv") {
    o = std::make_shared<CudnnConvolutionFwd>(p, n);
  } else if(n.type_ == "batchnorm") {
    o = std::make_shared<CudnnBatchNormFwd>(p, n);
  } else if(n.type_ == "fc") {
    o = std::make_shared<CudnnFcFwd>(p, n);
  } else if(n.type_ == "relu") {
    o = std::make_shared<CudnnReluFwd>(p, n);
  } else if(n.type_ == "maxpool") {
    o = std::make_shared<CudnnPoolingFwd>(p, n, CUDNN_POOLING_MAX);
  } else if(n.type_ == "avgpool") {
    o = std::make_shared<CudnnPoolingFwd>(p, n, CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING);
  } else if(n.type_ == "sum") {
    o = std::make_shared<CudnnSumFwd>(p, n);
  } else if(n.type_ == "softmax") {
    o = std::make_shared<CudnnSoftmaxFwd>(p, n);
  } else if(n.type_ == "reshape") {

    auto x = lower_tensor(p, n.inputs_.get("x"));
    auto y = n.outputs_.get("y");
    const std::string name(y->name_ + ".alias." + x->name_);

    p.tensors_[y] = std::make_shared<CudaTensor>(name, x->storage_, y->dims_,
                                                 CUDNN_TENSOR_NHWC);
    return;
  } else {
    printf("Cant emit forward operation for node type %s\n", n.type_.c_str());
    abort();
  }
  p.operations_.push_back(o);
}





std::shared_ptr<Program>
cudnn_inference(std::shared_ptr<Graph> g,
                int batch_size)
{
  printf("Generating program\n");

  auto p = std::make_shared<CudnnProgram>();

  p->ctx_ = std::make_shared<CudnnContext>();
  p->ctx_->init();

  for(const auto &n : g->inputs_) {
    p->inputs_.insert(lower_tensor(*p, n));
  }

  for(const auto &n : g->outputs_) {
    p->outputs_.insert(lower_tensor(*p, n));
  }

  for(const auto &n : g->nodes_) {
    generate_forward_operation(*p, *n);
  }

  for(const auto &n : g->nodes_) {
    printf(" == %s ======\n", n->type_.c_str());

    for(const auto &it : n->inputs_) {
      auto it2 = p->tensors_.find(it.second);

      printf("%15s: %s\n", it.first.c_str(),
             it2 != p->tensors_.end() ?
             it2->second->info().c_str() :
             it.second->info().c_str());
    }

    for(const auto &it : n->outputs_) {
      auto it2 = p->tensors_.find(it.second);

      printf("%15s: %s\n", it.first.c_str(),
             it2 != p->tensors_.end() ?
             it2->second->info().c_str() :
             it.second->info().c_str());
    }

  }

  printf("workspace needed: %zd\n", p->workspace_size_);

  p->ctx_->workspace_size_ = p->workspace_size_;

  chkCuda(cudaMalloc(&p->ctx_->workspace_, p->ctx_->workspace_size_));


  return p;
}


void
CudnnProgram::exec()
{
  printf("exec!\n");
  for(const auto &op : operations_) {
    op->exec();
  }
}




}
