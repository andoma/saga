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

#include <map>

#include "saga.h"
#include "tensor.h"
#include "context.h"

#include "cuda_common.h"
#include "cuda_tensor.h"


namespace saga {

int
CudaContext::init()
{
  cudaGetDevice(&deviceId_);

  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceId_);

  chkCuda(cudaStreamCreateWithFlags(&stream_,
                                    cudaStreamNonBlocking));

  printf("Device:%s (%d.%d) Concurrent:%s CanMapHostMem:%s id:%d\n",
         prop.name, prop.major, prop.minor,
         prop.concurrentKernels ? "yes":"no",
         prop.canMapHostMemory ? "yes":"no",
         deviceId_);


  chkCUDNN(cudnnCreate(&cudnn_));
  chkCUDNN(cudnnSetStream(cudnn_, stream_));

  chkCuda(cublasCreate(&cublas_));
  chkCuda(cublasSetStream(cublas_, stream_));
  chkCuda(cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH));
  return 0;
}


static
std::shared_ptr<Context> createCudaContext()
{
  auto ctx = std::make_shared<CudaContext>();

  if(ctx->init())
    return nullptr;

  return ctx;
}


static void __attribute__((constructor))
registerCudaContext(void)
{
  if(getenv("SAGA_DISABLE_CUDA"))
    return;
  registerContextFactory(ContextType::CUDA, &createCudaContext);
}



void
CudaProgram::infer(const std::shared_ptr<CudaOperation> &op)
{
  infer_operations_.push_back(op);
}

void
CudaProgram::train(const std::shared_ptr<CudaOperation> &op)
{
  train_operations_.push_back(op);
}

void
CudaProgram::bwd(const std::shared_ptr<CudaOperation> &op)
{
  bwd_operations_.insert(bwd_operations_.begin(), op);
}

void
CudaProgram::upd(const std::shared_ptr<CudaOperation> &op)
{
  upd_operations_.push_back(op);
}


std::shared_ptr<CudaTensor>
CudaProgram::resolveTensor_locked(std::shared_ptr<Tensor> src)
{
  if(src == nullptr)
    return nullptr;

  auto it = tensors_.find(src);
  if(it != tensors_.end()) {
    return it->second;
  }
  return nullptr;
}



std::shared_ptr<Tensor>
CudaProgram::resolveTensor(std::shared_ptr<Tensor> src)
{
  std::scoped_lock lock(ctx_->mutex_);
  return resolveTensor_locked(src);
}


cudnnTensorFormat_t
CudaProgram::tensorFormat(Tensor::DataType data_type)
{
  switch(tensor_layout_) {
  case TensorLayout::Auto:

    switch(data_type) {
    case Tensor::DataType::U8:
    case Tensor::DataType::HALF:
      return CUDNN_TENSOR_NHWC;
    default:
      return CUDNN_TENSOR_NCHW;
    }

  case TensorLayout::NHWC:
    return CUDNN_TENSOR_NHWC;

  case TensorLayout::NCHW:
    return CUDNN_TENSOR_NCHW;
  }
  abort();
}



std::shared_ptr<CudaTensor>
CudaProgram::lower_tensor(std::shared_ptr<Tensor> src, size_t rank)
{
  if(src == nullptr)
    return nullptr;

  auto it = tensors_.find(src);
  if(it != tensors_.end()) {
    return it->second;
  }

  Dims dims = src->dims_;

  if(rank) {
    while(dims.size() < rank)
      dims.insert(dims.begin(), 1);
  }

  auto t = std::make_shared<CudaTensor>(src->data_type_,
                                        dims, tensorFormat(src->data_type_),
                                        ctx_,
                                        src->name_);

  t->copyFromLocked(*src);
  tensors_[src] = t;
  return t;
}

std::shared_ptr<CudaTensor>
CudaProgram::lower_tensor_batch(std::shared_ptr<Tensor> src,
                                const CudaTensor &blueprint)
{
  if(src == nullptr)
    return nullptr;

  auto it = tensors_.find(src);
  if(it != tensors_.end()) {
    return it->second;
  }

  auto t = std::make_shared<CudaTensor>(src->data_type_, blueprint);
  t->copyFromLocked(*src);
  tensors_[src] = t;
  return t;
}

std::shared_ptr<CudaTensor>
CudaProgram::lower_tensor_batch(std::shared_ptr<Tensor> src,
                                cudnnTensorFormat_t tensor_format)
{
  if(src == nullptr)
    return nullptr;

  auto it = tensors_.find(src);
  if(it != tensors_.end()) {
    return it->second;
  }

  auto t = std::make_shared<CudaTensor>(src->data_type_,
                                        src->dims_.n(batch_size_),
                                        tensor_format,
                                        ctx_,
                                        src->name_);

  t->copyFromLocked(*src);
  tensors_[src] = t;
  return t;
}


std::shared_ptr<CudaTensor>
CudaProgram::lower_tensor_batch(std::shared_ptr<Tensor> src)
{
  return lower_tensor_batch(src, tensorFormat(src->data_type_));
}




void
CudaProgram::infer(long batches)
{
  if(batches == 0)
    return;

  issueOps(infer_pre_, 0);
  flipDoubleBufferedTensors();
  for(const auto &op : infer_operations_) op->load(*this, 0);
  cudaStreamSynchronize(ctx_->stream_);
  for(long i = 0; i < batches; i++) {

    for(const auto &op : infer_operations_) op->exec(*this);

    if(i < batches - 1)
      issueOps(infer_pre_, i + 1);
    if(i > 0)
      issueOps(infer_post_, i - 1);

    flipDoubleBufferedTensors();
    for(const auto &op : infer_operations_) op->load(*this, i + 1);
    cudaStreamSynchronize(ctx_->stream_);
  }
  issueOps(infer_post_, batches - 1);
}


void
CudaProgram::train(long batches)
{
  if(batches == 0)
    return;

  issueOps(train_pre_, 0);
  flipDoubleBufferedTensors();
  for(const auto &op : train_operations_) op->load(*this, 0);
  cudaStreamSynchronize(ctx_->stream_);

  for(long i = 0; i < batches; i++) {

    for(const auto &op : train_operations_) op->exec(*this);
    for(const auto &op : bwd_operations_)   op->exec(*this);
    for(const auto &op : upd_operations_)   op->exec(*this);

    if(i < batches - 1)
      issueOps(train_pre_, i + 1);
    if(i > 0)
      issueOps(train_post_, i - 1);

    flipDoubleBufferedTensors();
    if(i < batches - 1)
      for(const auto &op : train_operations_) op->load(*this, i + 1);
    cudaStreamSynchronize(ctx_->stream_);

    if(*(int *)check_result_) {
      mp_scaling_ *= 0.5;
      *(int *)check_result_ = 0;
    } else {
      mp_scaling_ *= 1.01;
    }
  }

  issueOps(train_post_, batches - 1);
}


void
CudaProgram::print() const
{
  std::scoped_lock lock(ctx_->mutex_);

  printf("\n\nInference:\n");
  for(const auto &op : infer_operations_) {
    op->print();
  }

  printf("\n\nTraining:\n");
  for(const auto &op : train_operations_) {
    op->print();
  }
  for(const auto &op : bwd_operations_) {
    op->print();
  }
  for(const auto &op : upd_operations_) {
    op->print();
  }
}

void
CudaProgram::debug(bool on)
{
  debug_ = on;
}


//------------------------------------------------------------------------

struct OpFactory {
  void (*mk_infer)(CudaProgram &p, const Node &n);
  void (*mk_train)(CudaProgram &p, const Node &n);
};

static std::map<std::string, OpFactory> *cuda_op_factories;

void
CudaRegisterOpFactory(const char *name,
                      void (*mk_infer)(CudaProgram &p, const Node &n),
                      void (*mk_train)(CudaProgram &p, const Node &n))
{
  if(!cuda_op_factories)
    cuda_op_factories = new  std::map<std::string, OpFactory>;

  (*cuda_op_factories)[name] = OpFactory{
    .mk_infer = mk_infer,
    .mk_train = mk_train
  };
}


static const OpFactory *
find_operation(const Node &n)
{
  if(!cuda_op_factories)
    return nullptr;
  auto it = cuda_op_factories->find(n.type_);
  if(it == cuda_op_factories->end())
    return nullptr;
  return &it->second;
}



struct CudaNodeTransform {
  CudaTransformType type;
  Nodes (*op)(CudaProgram &p, const Nodes &nodes);
};


static std::vector<CudaNodeTransform> *transforms;

void
CudaRegisterTransform(CudaTransformType type,
                      Nodes (*op)(CudaProgram &p, const Nodes &nodes))
{
  if(!transforms)
    transforms = new std::vector<CudaNodeTransform>;
  transforms->push_back(CudaNodeTransform{ .type = type, .op = op});
}


static std::vector<std::shared_ptr<Node>>
applyTransforms(CudaTransformType type,
                CudaProgram &p, std::vector<std::shared_ptr<Node>> nodes)
{
  for(auto const &cnt : *transforms) {
    if(type != cnt.type)
      continue;
    nodes = cnt.op(p, nodes);
  }
  return nodes;
}

static void
print_nodes(CudaProgram &p,
            const std::vector<std::shared_ptr<Node>> &nodes)
{
  std::vector<std::shared_ptr<Node>> r;

  for(size_t i = 0; i < nodes.size(); i++) {
    auto &n = nodes[i];

    printf("%s:\n", n->type_.c_str());

    for(const auto &t : n->inputs_) {
      auto l = p.resolveTensor_locked(t.second);
      printf("\t Input: %s: %s\n",
             t.first.c_str(), l ? l->info().c_str() : t.second->info().c_str());
    }

    for(const auto &t : n->outputs_) {
      auto l = p.resolveTensor_locked(t.second);
      printf("\tOutput: %s: %s\n",
             t.first.c_str(), l ? l->info().c_str() : t.second->info().c_str());
    }

    for(const auto &a : n->attributes_) {
      std::string value;

      if(auto v = std::get_if<int>(&a.second)) {
        value = std::to_string(*v);
      } else if(auto v = std::get_if<float>(&a.second)) {
        value = std::to_string(*v);
      } else if(auto v = std::get_if<bool>(&a.second)) {
        value = *v ? "true" : "false";
      } else if(std::get_if<std::vector<int>>(&a.second)) {
        value = "<vector>";
      } else {
        value = "?";
      }

      printf("\tAttrib: %s: %s\n",
             a.first.c_str(), value.c_str());
    }
  }
}



/**
 * If the network forward path splits into multiple nodes such as...
 *
 *                        +---+
 *                /=====> | B |
 *  +---+        /        +---+
 *  | A | ===== <
 *  +---+        \        +---+
 *                \=====> | C |
 *                        +---+
 *
 * ... results of backpropagation from B, C must be added together before
 * fed back into A.
 *
 * This code does so by adjusting the dx.beta to 1 for all nodes but
 * the first one (to be executed during backprop).
 *
 * dx.beta = 1 means that before writing a value the node will read
 * the current value and sum them together.
 */
static std::vector<std::shared_ptr<Node>>
compute_dx_beta(CudaProgram &p,
                const std::vector<std::shared_ptr<Node>> &nodes)
{
  std::vector<std::shared_ptr<Node>> r;
  std::unordered_set<std::shared_ptr<Tensor>> xset;

  for(ssize_t i = nodes.size() - 1; i >= 0; i--) {
    std::shared_ptr<Node> n = nodes[i];
    auto &x = n->inputs_["x"];
    if(x) {

      if(xset.find(x) == xset.end()) {
        // First contributing node. dx.beta = 0 (default)
        xset.insert(x);
      } else {
        // Other contributing nodes: Add to current value
        auto n2 = std::make_shared<Node>(*n);
        n2->attributes_["dx.beta"] = 1.0f;
        n = n2;
      }
    }
    r.insert(r.begin(), n);
  }
  return r;
}
REGISTER_CUDA_TRANSFORM(1000, CUDA_TRANSFORM_TRAINING, compute_dx_beta);


void
CudaProgram::addPrePostOp(std::shared_ptr<CudaTensor> t,
                          const BatchTensorAccess &a)
{
  auto op = CudaBatchAccessOp{.tensor_ = t, .fn_ = a.fn};

  if(a.phase == Phase::PRE) {
    op.prefetch_ = true;
    if(a.mode == Mode::INFER || a.mode == Mode::ALL)
      infer_pre_.push_back(op);

    if(a.mode == Mode::TRAIN || a.mode == Mode::ALL)
      train_pre_.push_back(op);

  } else {

    if(a.mode == Mode::INFER || a.mode == Mode::ALL)
      infer_post_.push_back(op);

    if(a.mode == Mode::TRAIN || a.mode == Mode::ALL)
      train_post_.push_back(op);
  }
}


void
CudaProgram::setupAccessors(const BatchTensorAccessors &accessors)
{
  for(const auto &a : accessors) {
    if(a.which != Which::VALUE)
      continue;

    auto src = a.tensor;
    auto dims = src->dims_.n(batch_size_);
    auto t = std::make_shared<CudaTensor>(src->data_type_, dims,
                                          tensorFormat(src->data_type_),
                                          ctx_, src->name_, 2);

    flips_.push_back(t->storage_);
    t->copyFromLocked(*src);
    tensors_[src] = t;
    addPrePostOp(t, a);
  }


  for(const auto &a : accessors) {
    if(a.which != Which::GRADIENT)
      continue;

    auto src = a.tensor;
    auto dims = src->dims_.n(batch_size_);
    auto g = std::make_shared<CudaTensor>(src->data_type_, dims,
                                          tensorFormat(src->data_type_),
                                          ctx_, src->name_, 2);
    flips_.push_back(g->storage_);

    auto t = lower_tensor_batch(src);
    t->grad_ = g;
    addPrePostOp(g, a);
  }

}



std::shared_ptr<Program>
CudaContext::createProgram(const Graph &g,
                           const ProgramConfig &pc,
                           const BatchTensorAccessors &accessors)
{
  std::scoped_lock lock(mutex_);

  auto p = std::make_shared<CudaProgram>(shared_from_this(),
                                          pc.tensor_layout,
                                          pc.batch_size,
                                          pc.initial_learning_rate);

  p->setupAccessors(accessors);

  auto nodes = applyTransforms(CUDA_TRANSFORM_ALL, *p, g.nodes_);

  if(pc.training) {
    auto train_nodes = applyTransforms(CUDA_TRANSFORM_TRAINING, *p, nodes);

    for(const auto &n : train_nodes) {
      auto op = find_operation(*n);
      if(op != NULL && op->mk_train) {
        op->mk_train(*p, *n);
      } else {
        fprintf(stderr, "Unable to create training operation for node %s\n",
                n->type_.c_str());
        n->print();
        exit(1);
      }
    }

    assert(p->infer_operations_.empty());
  }

  if(pc.inference) {
    auto infer_nodes = applyTransforms(CUDA_TRANSFORM_INFERENCE, *p, nodes);
    for(const auto &n : infer_nodes) {
      auto op = find_operation(*n);
      if(op != NULL && op->mk_infer) {
        op->mk_infer(*p, *n);
      } else {
        fprintf(stderr, "Unable to create inference operation for node %s\n",
                n->type_.c_str());
        n->print();
        exit(1);
      }
    }
  }
  p->allocWorkspace();


  return p;
}

}
