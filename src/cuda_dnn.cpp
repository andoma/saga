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



#define TENSOR_FORMAT CUDNN_TENSOR_NHWC

namespace saga {

class CudnnContext : public Context,
                     public std::enable_shared_from_this<CudnnContext> {
public:
  CudnnContext()
    : cudnn_(NULL)
    , cublas_(NULL)
  {}

  ~CudnnContext()
  {}

  int init();

  std::shared_ptr<Program> createProgram(const Graph &graph,
                                         ProgramType type,
                                         int batch_size);

  cudnnHandle_t cudnn_;
  cublasHandle_t cublas_;
};


int
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
  return 0;
}



std::shared_ptr<Context> createContext()
{
  auto ctx = std::make_shared<CudnnContext>();

  if(ctx->init())
    return nullptr;

  return ctx;
}


//------------------------------------------------------------------------

class CudnnOperation;

class CudnnProgram : public Program {
public:

  CudnnProgram(std::shared_ptr<CudnnContext> ctx)
    : ctx_(ctx)
    , workspace_(NULL)
    , workspace_size_(0)
    , workspace_requested_(0)
  {
  }

  ~CudnnProgram()
  {
    chkCuda(cudaFree(workspace_));
  }


  void exec();

  void requetstWorkspace(size_t size) {
    workspace_requested_ = std::max(workspace_requested_, size);
  }

  void allocWorkspace() {
    if(workspace_requested_ <= workspace_size_)
      return;
    workspace_size_ = workspace_requested_;
    chkCuda(cudaFree(workspace_));
    chkCuda(cudaMalloc(&workspace_, workspace_size_));
  }

  const std::shared_ptr<CudnnContext> ctx_;

  std::unordered_map<std::shared_ptr<Tensor>,
                     std::shared_ptr<CudaTensor>> tensors_;

  std::vector<std::shared_ptr<CudnnOperation>> operations_;

  void *workspace_;
  size_t workspace_size_;
  size_t workspace_requested_;
};




class CudnnOperation {
public:
  virtual ~CudnnOperation() {}
  virtual void exec(CudnnProgram &p) = 0;
};




static std::shared_ptr<CudaTensor>
lower_tensor(CudnnProgram &p, std::shared_ptr<Tensor> src,
             size_t dimensions = 0)
{
  if(src == nullptr)
    return nullptr;

  auto it = p.tensors_.find(src);
  if(it != p.tensors_.end()) {
    return it->second;
  }

  std::vector<int64_t> dims = src->dims_;

  if(dimensions) {
    while(dims.size() < dimensions)
      dims.insert(dims.begin(), 1);
  }

  auto t = std::make_shared<CudaTensor>(src->name_, src->data_type_,
                                        dims, TENSOR_FORMAT);

  t->copyFrom(*src);
  p.tensors_[src] = t;
  return t;
}



//------------------------------------------------------------------------

struct CudnnConvolutionFwd : public CudnnOperation {

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
    , b_(lower_tensor(p, n.inputs_.get("b"), 2))
    , y_(lower_tensor(p, n.outputs_.get("y")))
  {
    chkCUDNN(cudnnCreateFilterDescriptor(&filter_desc_));
    chkCUDNN(cudnnCreateConvolutionDescriptor(&conv_desc_));


    chkCUDNN(cudnnSetFilter4dDescriptor(filter_desc_,
                                        x_->type_,
                                        TENSOR_FORMAT,
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

    size_t workspace;
    chkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(ctx_->cudnn_,
                                                     x_->desc_,
                                                     filter_desc_,
                                                     conv_desc_,
                                                     y_->desc_,
                                                     conv_fwd_algo_,
                                                     &workspace));

    p.requetstWorkspace(workspace);
  }

  void exec(CudnnProgram &p) {
    float alpha = 1.0f, beta = 0.0f;
#if 0
    printf("x: %s\n", x_->info().c_str());
    printf("w: %s\n", w_->info().c_str());
    if(b_)
      printf("b: %s\n", b_->info().c_str());
    printf("y: %s\n", y_->info().c_str());
#endif

    chkCUDNN(cudnnConvolutionForward(ctx_->cudnn_, &alpha,
                                     x_->desc(),
                                     x_->deviceMem(),
                                     filter_desc_,
                                     w_->deviceMem(),
                                     conv_desc_,
                                     conv_fwd_algo_,
                                     p.workspace_, p.workspace_size_,
                                     &beta,
                                     y_->desc(),
                                     y_->deviceMem()));

    if(b_) {
      chkCUDNN(cudnnAddTensor(ctx_->cudnn_,
                              &alpha, b_->desc(), b_->deviceMem(),
                              &alpha, y_->desc(), y_->deviceMem()));
    }
  }

};



//------------------------------------------------------------------------

struct CudnnAddFwd : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, b_, y_;

  CudnnAddFwd(CudnnProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(lower_tensor(p, n.inputs_.get("x")))
    , b_(lower_tensor(p, n.inputs_.get("b"), x_ ? x_->dims_.size() : 0))
    , y_(lower_tensor(p, n.outputs_.get("y")))
  {
  }

  void exec(CudnnProgram &p) {
    float alpha = 1.0f, beta = 0.0f;
#if 0
    printf("x: %s\n", x_->info().c_str());
    printf("b: %s\n", b_->info().c_str());
    printf("y: %s\n", y_->info().c_str());
#endif
    // Use cudnnOpTensor
    chkCUDNN(cudnnTransformTensor(ctx_->cudnn_,
                                  &alpha,
                                  x_->desc(),
                                  x_->deviceMem(),
                                  &beta,
                                  y_->desc(),
                                  y_->deviceMem()));

    chkCUDNN(cudnnAddTensor(ctx_->cudnn_,
                            &alpha, b_->desc(), b_->deviceMem(),
                            &alpha, y_->desc(), y_->deviceMem()));
  }

};



//------------------------------------------------------------------------

struct CudnnBatchNormFwd : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, s_, b_, m_, v_, y_;
  const float epsilon_;

  CudnnBatchNormFwd(CudnnProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(lower_tensor(p, n.inputs_.get("x")))
    , s_(lower_tensor(p, n.inputs_.get("s"), 2))
    , b_(lower_tensor(p, n.inputs_.get("b"), 2))
    , m_(lower_tensor(p, n.inputs_.get("m"), 2))
    , v_(lower_tensor(p, n.inputs_.get("v"), 2))
    , y_(lower_tensor(p, n.outputs_.get("y")))
    , epsilon_(n.attributes_.get("epsilon", 1e-5f))
  {
  }

  void exec(CudnnProgram &p) {
    float alpha = 1.0f, beta = 0.0f;
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

struct CudnnReluFwd : public CudnnOperation {

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

  void exec(CudnnProgram &p) {
    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnActivationForward(ctx_->cudnn_, desc_,
                                    &alpha,
                                    x_->desc(), x_->deviceMem(),
                                    &beta,
                                    y_->desc(), y_->deviceMem()));
  }
};


//------------------------------------------------------------------------

struct CudnnPoolingFwd : public CudnnOperation {

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

  void exec(CudnnProgram &p) {
    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnPoolingForward(ctx_->cudnn_, desc_,
                                 &alpha,
                                 x_->desc(), x_->deviceMem(),
                                 &beta,
                                 y_->desc(), y_->deviceMem()));
  }
};


//------------------------------------------------------------------------

struct CudnnSumFwd : public CudnnOperation {

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

  void exec(CudnnProgram &p) {

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

struct CudnnGemmFwd : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, w_, b_, y_;
  const int n_;
  const int num_inputs_;
  const int num_outputs_;
  const int transW_;
  cudnnOpTensorDescriptor_t desc_;

  CudnnGemmFwd(CudnnProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(lower_tensor(p, n.inputs_.get("x")))
    , w_(lower_tensor(p, n.inputs_.get("w")))
    , b_(lower_tensor(p, n.inputs_.get("b"), 2))
    , y_(lower_tensor(p, n.outputs_.get("y")))
    , n_(x_->dims_[0])
    , num_inputs_(x_->dims_[1])
    , num_outputs_(y_->dims_[1])
    , transW_(n.attributes_.get("transW", 0))
  {
  }

  void exec(CudnnProgram &p) {

    float alpha = 1.0f, beta = 0.0f;
    __half halpha = 1.0f, hbeta = 0.0f;
    cublasOperation_t transA = CUBLAS_OP_T;
    cublasOperation_t transB = CUBLAS_OP_N;
    switch(x_->type_) {
    case CUDNN_DATA_FLOAT:
      chkCuda(cublasSgemm(ctx_->cublas_, transA, transB,
                          num_outputs_, n_, num_inputs_,
                          &alpha,
                          (const float *)w_->deviceMem(), num_inputs_,
                          (const float *)x_->deviceMem(), num_inputs_,
                          &beta,
                          (float *)y_->deviceMem(), num_outputs_));
      break;
    case CUDNN_DATA_HALF:
      chkCuda(cublasHgemm(ctx_->cublas_, transA, transB,
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
    if(b_) {
      chkCUDNN(cudnnAddTensor(ctx_->cudnn_,
                              &alpha, b_->desc(), b_->deviceMem(),
                              &alpha, y_->desc(), y_->deviceMem()));
    }
  }
};


//------------------------------------------------------------------------

struct CudnnSoftmaxFwd : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;

  CudnnSoftmaxFwd(CudnnProgram &p, const Node &n)
    : ctx_(p.ctx_)
    , x_(lower_tensor(p, n.inputs_.get("x")))
    , y_(lower_tensor(p, n.outputs_.get("y")))
  {
  }

  void exec(CudnnProgram &p) {

    float alpha = 1.0f, beta = 0.0f;

    chkCUDNN(cudnnSoftmaxForward(ctx_->cudnn_,
                                 CUDNN_SOFTMAX_ACCURATE,
                                 CUDNN_SOFTMAX_MODE_CHANNEL,
                                 &alpha,
                                 x_->desc(), x_->deviceMem(),
                                 &beta,
                                 y_->desc(), y_->deviceMem()));
  }
};

//------------------------------------------------------------------------

struct CudnnTransform : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> x_, y_;

  CudnnTransform(CudnnProgram &p,
                 std::shared_ptr<CudaTensor> x,
                 std::shared_ptr<CudaTensor> y)
    : ctx_(p.ctx_)
    , x_(x)
    , y_(y)
  {
  }

  void exec(CudnnProgram &p) {

    float alpha = 1.0f, beta = 0.0f;
#if 0
    printf("x: %s\n", x_->info().c_str());
    printf("y: %s\n", y_->info().c_str());
#endif
    chkCUDNN(cudnnTransformTensor(ctx_->cudnn_,
                                  &alpha,
                                  x_->desc(),
                                  x_->deviceMem(),
                                  &beta,
                                  y_->desc(),
                                  y_->deviceMem()));
  }
};

//------------------------------------------------------------------------

struct CudnnMathOp : public CudnnOperation {

  const std::shared_ptr<CudnnContext> ctx_;
  const std::shared_ptr<CudaTensor> a_, b_, c_;
  cudnnOpTensorDescriptor_t desc_;

  CudnnMathOp(CudnnProgram &p,
              std::shared_ptr<CudaTensor> a,
              std::shared_ptr<CudaTensor> b,
              std::shared_ptr<CudaTensor> c,
              cudnnOpTensorOp_t op)
    : ctx_(p.ctx_)
    , a_(a)
    , b_(b)
    , c_(c)
  {
    chkCUDNN(cudnnCreateOpTensorDescriptor(&desc_));
    chkCUDNN(cudnnSetOpTensorDescriptor(desc_,
                                        op, c->type_,
                                        CUDNN_PROPAGATE_NAN));
  }

  ~CudnnMathOp()
  {
    chkCUDNN(cudnnDestroyOpTensorDescriptor(desc_));
  }

  void exec(CudnnProgram &p) {

    float alpha = 1.0f, beta = 0.0f;
#if 0
    printf("a: %s\n", a_->info().c_str());
    printf("b: %s\n", b_->info().c_str());
    printf("c: %s\n", c_->info().c_str());
#endif

    chkCUDNN(cudnnOpTensor(ctx_->cudnn_,
                           desc_,
                           &alpha,
                           a_->desc(),
                           a_->deviceMem(),
                           &alpha,
                           b_->desc(),
                           b_->deviceMem(),
                           &beta,
                           c_->desc(),
                           c_->deviceMem()));
  }
};


//------------------------------------------------------------------------

struct CudnnPrintTensor : public CudnnOperation {

  const std::string prefix_;
  const std::shared_ptr<Tensor> x_;

  CudnnPrintTensor(const std::string &prefix, std::shared_ptr<Tensor> x)
    : prefix_(prefix)
    , x_(x)
  {
  }

  void exec(CudnnProgram &p) {
    x_->print(prefix_.c_str(), 4);
  }
};

//------------------------------------------------------------------------

struct CudnnPrintStatsTensor : public CudnnOperation {

  const std::string prefix_;
  const std::shared_ptr<Tensor> x_;

  CudnnPrintStatsTensor(const std::string &prefix, std::shared_ptr<Tensor> x)
    : prefix_(prefix)
    , x_(x)
  {
  }

  void exec(CudnnProgram &p) {
    printf("%s: %s\n", prefix_.c_str(), x_->info().c_str());
    x_->printStats(prefix_.c_str());
  }
};


//------------------------------------------------------------------------

struct CudnnHalt : public CudnnOperation {

  CudnnHalt()
  {
  }

  void exec(CudnnProgram &p) {
    fprintf(stderr, "Stopped by CudnnHalt\n");
    exit(1);
  }
};





//------------------------------------------------------------------------


static void
generate_forward_operation(CudnnProgram &p, const Node &n)
{
  std::shared_ptr<CudnnOperation> o;

  if(n.type_ == "conv") {
    o = std::make_shared<CudnnConvolutionFwd>(p, n);
  } else if(n.type_ == "add") {
    o = std::make_shared<CudnnMathOp>(p,
                                      lower_tensor(p, n.outputs_.get("x")),
                                      lower_tensor(p, n.outputs_.get("b")),
                                      lower_tensor(p, n.outputs_.get("y")),
                                      CUDNN_OP_TENSOR_ADD);
  } else if(n.type_ == "mul") {
    o = std::make_shared<CudnnMathOp>(p,
                                      lower_tensor(p, n.outputs_.get("x")),
                                      lower_tensor(p, n.outputs_.get("s")),
                                      lower_tensor(p, n.outputs_.get("y")),
                                      CUDNN_OP_TENSOR_MUL);
  } else if(n.type_ == "batchnorm") {
    o = std::make_shared<CudnnBatchNormFwd>(p, n);
  } else if(n.type_ == "gemm") {
    o = std::make_shared<CudnnGemmFwd>(p, n);
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
  } else if(n.type_ == "concat") {

    auto y = lower_tensor(p, n.outputs_.get("y"));
    auto element_offset = std::vector<int64_t>(y->dims_.size(), 0);
    const int axis = 1;
    for(const auto &xh : n.inputs_.getv("x")) {
      auto x = lower_tensor(p, xh);
      auto y2 = std::make_shared<CudaTensor>("concat.alias." + y->name_,
                                             y, x->dims_, element_offset);
      p.operations_.push_back(std::make_shared<CudnnTransform>(p, x, y2));

      element_offset[axis] += xh->dims_[axis];
    }
  } else if(n.type_ == "reshape" ||
            n.type_ == "dropout") {

    auto x = lower_tensor(p, n.inputs_.get("x"));
    auto y = n.outputs_.get("y");
    const std::string name(y->name_ + ".alias." + x->name_);

    p.tensors_[y] = std::make_shared<CudaTensor>(name, x->storage_,
                                                 y->dims_, TENSOR_FORMAT);

  } else {
    printf("Cant emit forward operation for node type %s\n", n.type_.c_str());
    abort();
  }

  if(o)
    p.operations_.push_back(o);

  return;

  p.operations_.push_back(std::make_shared<CudnnPrintStatsTensor>(n.type_,
                                                                  lower_tensor(p, n.outputs_.get("y"))));
}





std::shared_ptr<Program>
cudnn_inference(const Graph &g, int batch_size,
                std::shared_ptr<CudnnContext> ctx)
{
  auto p = std::make_shared<CudnnProgram>(ctx);

  for(const auto &n : g.inputs_) {
    p->inputs_.insert(lower_tensor(*p, n));
  }

  for(const auto &n : g.nodes_) {
    generate_forward_operation(*p, *n);
  }

  for(const auto &n : g.outputs_) {
    p->outputs_.insert(lower_tensor(*p, n));
  }


#if 0
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
#endif
  return p;
}

void
CudnnProgram::exec()
{
  allocWorkspace();
  for(const auto &op : operations_) {
    op->exec(*this);
  }
}


std::shared_ptr<Program>
CudnnContext::createProgram(const Graph &graph,
                            ProgramType type,
                            int batch_size)
{
  return cudnn_inference(graph, batch_size, shared_from_this());
}





}
