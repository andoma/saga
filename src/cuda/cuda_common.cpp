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
#include <map>

#include "saga.h"
#include "tensor.h"
#include "context.h"

#include "cuda_common.h"
#include "cuda_tensor.h"
#include "cuda_analysis.h"

namespace saga {

int
CudaContext::init()
{
#ifdef HAVE_NVIDIA_ML
  nvmlInit();
#endif

  cudaGetDevice(&deviceId_);

  struct cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceId_);

  chkCuda(cudaStreamCreateWithFlags(&stream_,
                                    cudaStreamNonBlocking));

  char pciid[32];
  cudaDeviceGetPCIBusId(pciid, sizeof(pciid), deviceId_);

  printf("Device:%s (%d.%d) Concurrent:%s CanMapHostMem:%s id:%d at %s\n",
         prop.name, prop.major, prop.minor,
         prop.concurrentKernels ? "yes":"no",
         prop.canMapHostMemory ? "yes":"no",
         deviceId_, pciid);


  chkCUDNN(cudnnCreate(&cudnn_));
  chkCUDNN(cudnnSetStream(cudnn_, stream_));

  cublasCreate(&cublas_);
  cublasSetStream(cublas_, stream_);
  cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH);

#ifdef HAVE_NVIDIA_ML
  nvmlDeviceGetHandleByPciBusId_v2(pciid, &nvmldev_);
#endif
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
CudaProgram::fwd(const std::shared_ptr<CudaOperation> &op)
{
  fwd_operations_.push_back(op);
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
    auto t = it->second;
    exported_storage_.push_back(t->storage_);
    return t;
  }
  return nullptr;
}

std::shared_ptr<Tensor>
CudaProgram::resolveTensor(std::shared_ptr<Tensor> src)
{
  std::scoped_lock lock(ctx_->mutex_);
  return resolveTensor_locked(src);
}



std::shared_ptr<CudaTensor>
CudaProgram::resolveTensorGradient_locked(std::shared_ptr<Tensor> src)
{
  if(src == nullptr)
    return nullptr;

  auto it = tensors_.find(src);
  if(it != tensors_.end()) {
    auto t = it->second->makeSharedGrad();
    exported_storage_.push_back(t->storage_);
    return t;
  }
  return nullptr;
}



std::shared_ptr<Tensor>
CudaProgram::resolveTensorGradient(std::shared_ptr<Tensor> src)
{
  std::scoped_lock lock(ctx_->mutex_);
  return resolveTensorGradient_locked(src);
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
                                        dims,
                                        rank == 2 ? CUDNN_TENSOR_NCHW :
                                        tensorFormat(src->data_type_),
                                        ctx_,
                                        src->name_);

  t->copyFromLocked(*src, 0);
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
  t->copyFromLocked(*src, 0);
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

  t->copyFromLocked(*src, 0);
  tensors_[src] = t;
  return t;
}


std::shared_ptr<CudaTensor>
CudaProgram::lower_tensor_batch(std::shared_ptr<Tensor> src)
{
  if(src == nullptr)
    return nullptr;
  return lower_tensor_batch(src, tensorFormat(src->data_type_));
}



void
CudaProgram::setupTensorStorage(std::shared_ptr<CudaMemoryLayout> cml)
{
  if(!cml)
    return;

  void *base = tensor_mem_.ptr();
  for(const auto &kv : cml->table_) {
    void *ptr = (void *)((char *)base + kv.second);
    //    printf("Tensor T%d gets %p\n", kv.first->id_, ptr);
    kv.first->setTmpMem(ptr);
  }
}



bool
CudaProgram::runOps(const CudaOps &ops, long batch)
{
  for(const auto &op : ops) {
    const char *err = op->exec(*this, batch);
    if(err) {
      fprintf(stderr, "\nOp %s failed: %s\n", op->name().c_str(), err);
      op->print(true);
      return false;
    }
  }
  return true;
}

void
CudaProgram::progress(const char *what, long i, long batches,
                      float mp_scaling, int64_t start_time)
{
  if(!print_progress_)
    return;
  time_t now = time(NULL);
  if(now == print_progress_ts_)
    return;
  print_progress_ts_ = now;

  printf("\033[K");

  size_t memfree = 0, memtotal = 0;
  cudaMemGetInfo(&memfree, &memtotal);

  printf("%-5s | Batch: %4ld/%-4ld | MemUse: %5zu/%-5zu MB",
         what, i, batches,
         (memtotal - memfree) / (1024 * 1024),
         memtotal / (1024 * 1024));

#ifdef HAVE_NVIDIA_ML
  nvmlUtilization_t util = {};
  if(!nvmlDeviceGetUtilizationRates(ctx_->nvmldev_, &util)) {
    printf(" | GpuUse: %3d%% MemUse: %3d%%",
           util.gpu, util.memory);
  }
#endif
  if(isfinite(mp_scaling)) {
    printf(" | MPS: %1.1e", mp_scaling);
  }

  if(i > 0) {
    float time_per_batch = (Now() - start_time) / i;
    printf(" | Bat/s: %3.2f", 1e6 / time_per_batch);
  }

  printf("\r");
  fflush(stdout);
  print_progress_pending_nl_ = true;
}

void
CudaProgram::progressDone(void)
{
  if(!print_progress_pending_nl_)
    return;
  printf("\n");
  print_progress_pending_nl_ = false;
}



ExecResult
CudaProgram::infer(long batches)
{
  if(batches == 0)
    return ExecResult::OK;

  if(stop_check_ && stop_check_())
    return ExecResult::STOPPED;

  finalize();

  setupTensorStorage(infer_memory_layout_);

  issueBatchAccessOps(infer_pre_, 0);

  flipDoubleBufferedTensors();
  if(!runOps(load_operations_, 0)) {
    return ExecResult::ERROR;
  }

  cudaStreamSynchronize(ctx_->stream_);
  int64_t start = Now();
  for(long i = 0; i < batches; i++) {

    if(!runOps(infer_operations_, i)) {
      progressDone();
      return ExecResult::ERROR;
    }
    if(i < batches - 1)
      issueBatchAccessOps(infer_pre_, i + 1);
    if(i > 0)
      issueBatchAccessOps(infer_post_, i - 1);

    flipDoubleBufferedTensors();
    if(i < batches - 1) {
      if(!runOps(load_operations_, i + 1)) {
        progressDone();
        return ExecResult::ERROR;
      }
    }

    if(stop_check_ && stop_check_()) {
      progressDone();
      return ExecResult::STOPPED;
    }

    progress("Test", i, batches, NAN, start);
    cudaStreamSynchronize(ctx_->stream_);
  }
  issueBatchAccessOps(infer_post_, batches - 1);
  progressDone();
  return ExecResult::OK;
}


ExecResult
CudaProgram::train(long batches)
{
  if(batches == 0)
    return ExecResult::OK;

  if(stop_check_ && stop_check_())
    return ExecResult::STOPPED;

  finalize();

  setupTensorStorage(train_memory_layout_);

  issueBatchAccessOps(train_pre_, 0);
  flipDoubleBufferedTensors();
  if(!runOps(load_operations_, 0)) {
    return ExecResult::ERROR;
  }

  cudaStreamSynchronize(ctx_->stream_);

  float current_mp_scaling = NAN;

  int64_t start = Now();

  for(long i = 0; i < batches; i++) {

    if(!runOps(train_operations_, i)) {
      progressDone();
      return ExecResult::ERROR;
    }

    if(i < batches - 1)
      issueBatchAccessOps(train_pre_, i + 1);
    if(i > 0)
      issueBatchAccessOps(train_post_, i - 1);

    flipDoubleBufferedTensors();
    if(i < batches - 1) {
      if(!runOps(load_operations_, i + 1)) {
        progressDone();
        return ExecResult::ERROR;
      }
    }

    if(stop_check_ && stop_check_()) {
      progressDone();
      return ExecResult::STOPPED;
    }

    progress("Train", i, batches, current_mp_scaling, start);
    cudaStreamSynchronize(ctx_->stream_);

    if(mp_enabled_) {
      if(*(int *)check_result_) {
        mp_scaling_ *= 0.5;
        *(int *)check_result_ = 0;
      } else {
        mp_scaling_ *= 1.01;
      }
      current_mp_scaling = mp_scaling_;
    }
  }

  issueBatchAccessOps(train_post_, batches - 1);
  progressDone();
  return ExecResult::OK;
}

std::string
CudaOperation::str() const
{
  std::stringstream ss;
  const char *sep = "";

  for(auto const &t : getOutputs()) {
    ss << sep << t->shortname();
    sep = ", ";
  }

  ss << " = " << name() << "(";
  sep = "";
  for(auto const &t : getInputs()) {
    ss << sep << t->shortname();
    sep = ", ";
  }
  auto inf = info();
  if(inf.size())
    ss << ", " << inf;

  ss << ")";
  return ss.str();
}


void
CudaOperation::print(bool full) const
{
  auto inputs = getInputs();
  auto outputs = getOutputs();

  if(full) {

    printf("OP: %s %s\n", name().c_str(), info().c_str());
    for(auto const &t : inputs) {
      if(t)
        printf("\tI: %s\n", t->info().c_str());
    }
    for(auto const &t : outputs) {
      if(t)
        printf("\tO: %s\n", t->info().c_str());
    }
  } else {
    printf("%s\n", str().c_str());
  }
}

void
CudaProgram::print(bool detailed) const
{
  std::scoped_lock lock(ctx_->mutex_);
  printf("\nInference: (%zd ops)\n", infer_operations_.size());
  int index = 0;
  for(const auto &op : infer_operations_) {
    if(detailed) {
      op->print(true);
    } else {
      printf("%3d: ", index);
      op->print();
    }
    index++;
  }

  printf("\nTraining: (%zd ops):\n", train_operations_.size());
  index = 0;
  for(const auto &op : train_operations_) {
    if(detailed) {
      op->print(true);
    } else {
      printf("%3d: ", index);
      op->print();
    }
    index++;
  }
}

void
CudaProgram::debug(bool on)
{
  debug_ = on;
}


//------------------------------------------------------------------------

struct OpFactory {
  const char *(*setup)(CudaProgram &p, const Node &n, bool training);
};

static std::map<std::string, OpFactory> *cuda_op_factories;

void
CudaRegisterOpFactory(const char *name,
                      const char *(*setup)(CudaProgram &p, const Node &n,
                                           bool train))
{
  if(!cuda_op_factories)
    cuda_op_factories = new std::map<std::string, OpFactory>;

  (*cuda_op_factories)[name] = OpFactory{.setup = setup };
}



static const char *
no_setup(CudaProgram &p, const Node &n, bool train)
{
  return "operation does not exist";
}

static const OpFactory no_op = {
  .setup = no_setup
};

static const OpFactory *
find_operation(const Node &n)
{
  if(!cuda_op_factories)
    return &no_op;
  auto it = cuda_op_factories->find(n.type_);
  if(it == cuda_op_factories->end())
    return &no_op;
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


static Nodes
applyTransforms(CudaTransformType type, CudaProgram &p, const Nodes &nodes)
{
  auto copy = nodes;
  for(auto const &cnt : *transforms) {
    if(type != cnt.type)
      continue;
    copy = cnt.op(p, copy);
  }
  return copy;
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
static Nodes
compute_dx_beta(CudaProgram &p, const Nodes &nodes)
{
  Nodes r;
  std::unordered_set<std::shared_ptr<Tensor>> xset;

  for(ssize_t i = nodes.size() - 1; i >= 0; i--) {
    std::shared_ptr<Node> n = nodes[i];

    for(const auto &it : n->inputs_) {
      const auto &name = it.first;
      if(name[0] != 'x')
        continue;

      auto &x = it.second;

      if(xset.find(x) == xset.end()) {
        // First contributing node. dx.beta = 0 (default)
        xset.insert(x);
      } else {
        // Other contributing nodes: Add to current value
        auto n2 = std::make_shared<Node>(*n);
        n2->attributes_["d" + name + ".beta"] = 1.0f;
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
                          std::shared_ptr<CudaTensorStorageDoubleBuffered> s,
                          const BatchTensorAccess &a)
{
  auto op = CudaBatchAccessOp{.tensor_ = t, .storage_ = s, .fn_ = a.fn};

  if(a.phase == Phase::PRE) {
    op.prefetch_ = true;
    if(a.mode == Mode::INFER || a.mode == Mode::ALL)
      infer_pre_.push_back(op);

    if(a.mode == Mode::TRAIN || a.mode == Mode::ALL)
      train_pre_.push_back(op);

  } else {

    exported_storage_.push_back(s);

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

    auto fmt = tensorFormat(src->data_type_);
    auto s = std::make_shared<CudaTensorStorageDoubleBuffered>(src->data_type_,
                                                               dims, fmt, ctx_);
    auto t = std::make_shared<CudaTensor>(s, dims, fmt);

    flips_.push_back(s);
    t->copyFromLocked(*src);
    tensors_[src] = t;
    addPrePostOp(t, s, a);
  }


  for(const auto &a : accessors) {
    if(a.which != Which::GRADIENT)
      continue;

    auto src = a.tensor;
    auto dims = src->dims_.n(batch_size_);

    auto fmt = tensorFormat(src->data_type_);
    auto s = std::make_shared<CudaTensorStorageDoubleBuffered>(src->data_type_,
                                                               dims, fmt, ctx_);
    auto g = std::make_shared<CudaTensor>(s, dims, fmt);
    flips_.push_back(s);

    auto t = lower_tensor_batch(src);
    t->grad_ = g;
    addPrePostOp(g, s, a);
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
                                         pc.initial_learning_rate,
                                         pc.stop_check,
                                         pc.show_progress);

  p->setupAccessors(accessors);

  auto nodes = applyTransforms(CUDA_TRANSFORM_ALL, *p, g.nodes_);

  if(pc.training) {
    auto train_nodes = applyTransforms(CUDA_TRANSFORM_TRAINING, *p, nodes);

    for(const auto &n : train_nodes) {
      const char *err = find_operation(*n)->setup(*p, *n, true);
      if(err) {
        fprintf(stderr, "Unable to create training operation for %s -- %s\n",
                n->type_.c_str(), err);
        n->print();
        exit(1);
      }
    }

    assert(p->infer_operations_.empty());

    p->train_operations_.insert(p->train_operations_.end(),
                                p->fwd_operations_.begin(),
                                p->fwd_operations_.end());
    p->train_operations_.insert(p->train_operations_.end(),
                                p->bwd_operations_.begin(),
                                p->bwd_operations_.end());
    p->train_operations_.insert(p->train_operations_.end(),
                                p->upd_operations_.begin(),
                                p->upd_operations_.end());

    p->fwd_operations_.clear();
    p->bwd_operations_.clear();
    p->upd_operations_.clear();

  }

  if(pc.inference) {
    auto infer_nodes = applyTransforms(CUDA_TRANSFORM_INFERENCE, *p, nodes);
    for(const auto &n : infer_nodes) {
      const char *err = find_operation(*n)->setup(*p, *n, false);
      if(err) {
        fprintf(stderr, "Unable to create inferenece operation for %s -- %s\n",
                n->type_.c_str(), err);
        n->print();
        exit(1);
      }
    }
  }

  for(const auto &kv : p->load_map_) {
    p->load_operations_.push_back(kv.second);
  }

  p->load_map_.clear();
  return p;
}

void
CudaProgram::finalize()
{
  if(finalized_)
    return;
  finalized_ = true;

  if(train_operations_.size()) {
    train_operations_ = reduceLiveranges(train_operations_, exported_storage_);
    train_memory_layout_ = memoryLayout(train_operations_, exported_storage_);
    tensor_mem_.request(train_memory_layout_->size_);
  }

  if(infer_operations_.size()) {
    infer_memory_layout_ = memoryLayout(infer_operations_, exported_storage_);
    tensor_mem_.request(infer_memory_layout_->size_);
  }

  tensor_mem_.alloc();

  ctx_->workspace_.alloc();
}


void
CudaContext::print()
{
  size_t memfree = 0, memtotal = 0;
  cudaMemGetInfo(&memfree, &memtotal);
  printf("   Free memory: %zd kbyte\n", memfree / 1024);
  printf("  Total memory: %zd kbyte\n", memtotal / 1024);
}

bool
CudaProgram::dumpGraphFromOps(const char *path, const CudaOps &ops)
{
  FILE *fp = fopen(path, "w");
  if(fp == NULL) {
    perror("fopen");
    return false;
  }

  std::unordered_set<std::shared_ptr<CudaTensorStorage>> storage;

  fprintf(fp, "digraph CudaOps {\n");


  int i = 0;
  for(auto &op : ops) {

    fprintf(fp, "node [shape=circle,label=\"%s\"]; O%d;\n",
            op->name().c_str(), i);
    i++;

    for(auto const &t : op->getInputs()) {
      storage.insert(t->storage_);
    }
    for(auto const &t : op->getOutputs()) {
      storage.insert(t->storage_);
    }
  }


  for(auto const &s : storage) {
    int id = s->id_;
    fprintf(fp, "node [shape=box,label=\"T%d\"]; T%d;\n",
            id, id);
  }
  fprintf(fp, "\n");


  i = 0;
  for(auto &op : ops) {

    for(auto const &t : op->getInputs()) {
      fprintf(fp, "T%d->O%d;\n", t->storage_->id_, i);

    }
    for(auto const &t : op->getOutputs()) {
      fprintf(fp, "O%d->T%d;\n", i, t->storage_->id_);
    }
    i++;
  }

  fprintf(fp, "overlap=false\n");

  fprintf(fp, "}\n");
  fclose(fp);
  return true;
}


bool
CudaProgram::dumpGraph(const char *path)
{
  return dumpGraphFromOps(path, train_operations_);
}

}
