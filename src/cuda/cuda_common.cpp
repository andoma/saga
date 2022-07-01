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

#include "saga.hpp"
#include "tensor.hpp"
#include "context.hpp"

#include "cuda_common.hpp"
#include "cuda_tensor.hpp"
#include "cuda_analysis.hpp"

namespace saga {

int
CudaContext::init()
{
#ifdef HAVE_NVIDIA_ML
    nvmlInit();
#endif

    cudaGetDevice(&m_deviceId);

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, m_deviceId);

    chkCuda(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));

    char pciid[32];
    cudaDeviceGetPCIBusId(pciid, sizeof(pciid), m_deviceId);

    m_tensor_cores = prop.major >= 7;

    chkCUDNN(cudnnCreate(&m_cudnn));
    chkCUDNN(cudnnSetStream(m_cudnn, m_stream));

    cublasCreate(&m_cublas);
    cublasSetStream(m_cublas, m_stream);
    if(m_tensor_cores)
        cublasSetMathMode(m_cublas, CUBLAS_TENSOR_OP_MATH);

#ifdef HAVE_NVIDIA_ML
    nvmlDeviceGetHandleByPciBusId_v2(pciid, &m_nvmldev);
#endif
    return 0;
}

static std::shared_ptr<Context>
createCudaContext()
{
    auto ctx = std::make_shared<CudaContext>();

    if(ctx->init())
        return nullptr;

    return ctx;
}

static void __attribute__((constructor)) registerCudaContext(void)
{
    if(getenv("SAGA_DISABLE_CUDA"))
        return;
    registerContextFactory(ContextType::CUDA, &createCudaContext);
}

void
CudaProgram::infer(const std::shared_ptr<CudaOperation> &op)
{
    m_infer_operations.push_back(op);
}

void
CudaProgram::fwd(const std::shared_ptr<CudaOperation> &op)
{
    m_fwd_operations.push_back(op);
}

void
CudaProgram::bwd(const std::shared_ptr<CudaOperation> &op)
{
    m_bwd_operations.insert(m_bwd_operations.begin(), op);
}

void
CudaProgram::upd(const std::shared_ptr<CudaOperation> &op)
{
    m_upd_operations.push_back(op);
}

std::shared_ptr<CudaTensor>
CudaProgram::resolveTensor_locked(std::shared_ptr<Tensor> src)
{
    if(src == nullptr)
        return nullptr;

    auto it = m_tensors.find(src);
    if(it != m_tensors.end()) {
        auto t = it->second;
        m_exported_storage.push_back(t->m_storage);
        return t;
    }
    return nullptr;
}

std::shared_ptr<Tensor>
CudaProgram::resolveTensor(std::shared_ptr<Tensor> src)
{
    std::scoped_lock lock(m_ctx->m_mutex);
    return resolveTensor_locked(src);
}

std::shared_ptr<CudaTensor>
CudaProgram::resolveTensorGradient_locked(std::shared_ptr<Tensor> src)
{
    if(src == nullptr)
        return nullptr;

    auto it = m_tensors.find(src);
    if(it != m_tensors.end()) {
        auto t = it->second->makeSharedGrad();
        m_exported_storage.push_back(t->m_storage);
        return t;
    }
    return nullptr;
}

std::shared_ptr<Tensor>
CudaProgram::resolveTensorGradient(std::shared_ptr<Tensor> src)
{
    std::scoped_lock lock(m_ctx->m_mutex);
    return resolveTensorGradient_locked(src);
}

cudnnTensorFormat_t
CudaProgram::tensorFormat(Tensor::DataType data_type, int rank)
{
    if(rank < 4)
        return CUDNN_TENSOR_NCHW;

    switch(m_tensor_layout) {
    case TensorLayout::Auto:

        switch(data_type) {
        case Tensor::DataType::U8:
        case Tensor::DataType::HALF:
            if(m_ctx->m_tensor_cores)
                return CUDNN_TENSOR_NHWC;
            // FALLTHRU
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
CudaProgram::lower_tensor(std::shared_ptr<Tensor> src, size_t minimum_rank)
{
    if(src == nullptr)
        return nullptr;

    auto it = m_tensors.find(src);
    if(it != m_tensors.end()) {
        return it->second;
    }

    Dims dims = src->dims_;

    if(minimum_rank) {
        while(dims.size() < minimum_rank) dims.insert(dims.begin(), 1);
    }

    dims = dims.n(m_batch_size);

    auto t = std::make_shared<CudaTensor>(
        src->data_type_, dims, tensorFormat(src->data_type_, dims.size()),
        m_ctx, src->name_);

    t->copyFromLocked(*src, 0);
    m_tensors[src] = t;
    return t;
}

std::shared_ptr<CudaTensor>
CudaProgram::lower_tensor(std::shared_ptr<Tensor> src,
                          const CudaTensor &blueprint)
{
    if(src == nullptr)
        return nullptr;

    auto it = m_tensors.find(src);
    if(it != m_tensors.end()) {
        return it->second;
    }

    auto t = std::make_shared<CudaTensor>(src->data_type_, blueprint);
    t->copyFromLocked(*src, 0);
    m_tensors[src] = t;
    return t;
}

std::shared_ptr<CudaTensor>
CudaProgram::lower_tensor(std::shared_ptr<Tensor> src,
                          cudnnTensorFormat_t tensor_format)
{
    if(src == nullptr)
        return nullptr;

    auto it = m_tensors.find(src);
    if(it != m_tensors.end()) {
        return it->second;
    }

    auto t = std::make_shared<CudaTensor>(src->data_type_,
                                          src->dims_.n(m_batch_size),
                                          tensor_format, m_ctx, src->name_);

    t->copyFromLocked(*src, 0);
    m_tensors[src] = t;
    return t;
}

void
CudaProgram::setupTensorStorage(std::shared_ptr<CudaMemoryLayout> cml)
{
    if(!cml)
        return;

    void *base = m_tensor_mem.ptr();
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

static void
updateMemUsage(UI &ui)
{
    size_t memfree = 0, memtotal = 0;
    cudaMemGetInfo(&memfree, &memtotal);
    ui.updateMemUsage(memtotal - memfree, memtotal);
}

ExecResult
CudaProgram::infer(long batches, const TensorBatchCallback &pre,
                   const TensorBatchCallback &post)
{
    if(batches == 0)
        return ExecResult::OK;

    if(m_stop_check && m_stop_check())
        return ExecResult::STOPPED;

    finalize();

    setupTensorStorage(m_infer_memory_layout);

    run_batched_tensor_callbacks(pre, false, 0, m_pre_batched_tensors);

    flipDoubleBufferedTensors();
    if(!runOps(m_load_operations, 0)) {
        return ExecResult::ERROR;
    }

    if(m_ui) {
        m_ui->updateBatchInfo(UI::INFER, m_batch_size, batches, m_epoch);
    }

    cudaStreamSynchronize(m_ctx->m_stream);
    for(long i = 0; i < batches; i++) {
        if(!runOps(m_infer_operations, i)) {
            return ExecResult::ERROR;
        }
        if(i < batches - 1) {
            run_batched_tensor_callbacks(pre, false, i + 1,
                                         m_pre_batched_tensors);
        }
        if(i > 0) {
            run_batched_tensor_callbacks(post, false, i - 1,
                                         m_post_batched_tensors);
        }

        flipDoubleBufferedTensors();
        if(i < batches - 1) {
            if(!runOps(m_load_operations, i + 1)) {
                return ExecResult::ERROR;
            }
        }

        if(m_stop_check && m_stop_check()) {
            return ExecResult::STOPPED;
        }

        m_total_inferred += m_batch_size;

        if(m_ui) {
            m_ui->updateProgress(i, m_total_inferred);
            updateMemUsage(*m_ui);
        }

        cudaStreamSynchronize(m_ctx->m_stream);
    }
    run_batched_tensor_callbacks(post, false, batches - 1,
                                 m_post_batched_tensors);
    return ExecResult::OK;
}

ExecResult
CudaProgram::train(long batches, const TensorBatchCallback &pre,
                   const TensorBatchCallback &post)
{
    if(batches == 0)
        return ExecResult::OK;

    if(m_stop_check && m_stop_check())
        return ExecResult::STOPPED;

    finalize();

    setupTensorStorage(m_train_memory_layout);

    run_batched_tensor_callbacks(pre, true, 0, m_pre_batched_tensors);

    flipDoubleBufferedTensors();
    if(!runOps(m_load_operations, 0)) {
        return ExecResult::ERROR;
    }

    if(m_ui) {
        m_ui->updateBatchInfo(UI::TRAIN, m_batch_size, batches, m_epoch);
    }
    m_epoch++;
    cudaStreamSynchronize(m_ctx->m_stream);

    for(long i = 0; i < batches; i++) {
        if(!runOps(m_train_operations, i)) {
            return ExecResult::ERROR;
        }

        if(i < batches - 1) {
            run_batched_tensor_callbacks(pre, true, i + 1,
                                         m_pre_batched_tensors);
        }
        if(i > 0) {
            run_batched_tensor_callbacks(post, true, i - 1,
                                         m_post_batched_tensors);
        }
        flipDoubleBufferedTensors();
        if(i < batches - 1) {
            if(!runOps(m_load_operations, i + 1)) {
                return ExecResult::ERROR;
            }
        }

        if(m_stop_check && m_stop_check()) {
            return ExecResult::STOPPED;
        }

        if(m_ui) {
            m_ui->updateProgress(i, m_total_trained);
            updateMemUsage(*m_ui);
        }

        m_total_trained += m_batch_size;

        cudaStreamSynchronize(m_ctx->m_stream);

        if(m_mp_enabled) {
            const int check_result = *(int *)m_check_result;
            if(check_result) {
                if(check_result & 2) {
                    printf("\nNAN detected\n");
                    return ExecResult::ERROR;
                }
                m_mp_scaling *= 0.5;
                *(int *)m_check_result = 0;
            } else {
                m_mp_scaling *= 1.01;
            }
            if(m_ui) {
                m_ui->updateMpScaling(m_mp_scaling);
            }
        }
    }

    run_batched_tensor_callbacks(post, true, batches - 1,
                                 m_post_batched_tensors);
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
    std::scoped_lock lock(m_ctx->m_mutex);
    printf("\nInference: (%zd ops)\n", m_infer_operations.size());
    int index = 0;
    for(const auto &op : m_infer_operations) {
        printf("#%3d: ", index);
        op->print(detailed);
        index++;
    }

    printf("\nTraining: (%zd ops):\n", m_train_operations.size());
    index = 0;
    for(const auto &op : m_train_operations) {
        printf("#%3d: ", index);
        op->print(detailed);
        index++;
    }
}

void
CudaProgram::debug(bool on)
{
    m_debug = on;
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

    (*cuda_op_factories)[name] = OpFactory{.setup = setup};
}

static const char *
no_setup(CudaProgram &p, const Node &n, bool train)
{
    return "operation does not exist";
}

static const OpFactory no_op = {.setup = no_setup};

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
    transforms->push_back(CudaNodeTransform{.type = type, .op = op});
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
print_nodes(CudaProgram &p, const std::vector<std::shared_ptr<Node>> &nodes)
{
    std::vector<std::shared_ptr<Node>> r;

    for(size_t i = 0; i < nodes.size(); i++) {
        auto &n = nodes[i];

        printf("%s:\n", n->type_.c_str());

        for(const auto &t : n->inputs_) {
            auto l = p.resolveTensor_locked(t.second);
            printf("\t Input: %s: %s\n", t.first.c_str(),
                   l ? l->info().c_str() : t.second->info().c_str());
        }

        for(const auto &t : n->outputs_) {
            auto l = p.resolveTensor_locked(t.second);
            printf("\tOutput: %s: %s\n", t.first.c_str(),
                   l ? l->info().c_str() : t.second->info().c_str());
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

            printf("\tAttrib: %s: %s\n", a.first.c_str(), value.c_str());
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
CudaProgram::addPrePostOp(std::shared_ptr<Tensor> &high,
                          std::shared_ptr<CudaTensor> &low,
                          std::shared_ptr<CudaTensorStorageDoubleBuffered> &s,
                          BTM mode)
{
    auto op = CudaBatchAccessOp{high, low, s, mode};

    if(!(mode & BTM::POST)) {
        m_pre_batched_tensors.push_back(op);

    } else {
        m_exported_storage.push_back(s);
        m_post_batched_tensors.push_back(op);
    }
}

void
CudaProgram::setupBatchedTensors(const BatchedTensors &bts)
{
    // Do values first and gradients in a second pass to make sure lowering is
    // correct
    for(const auto &bt : bts) {
        if(!!(bt.second & BTM::GRADIENT))
            continue;

        auto src = bt.first;
        auto dims = src->dims_.n(m_batch_size);

        auto fmt = tensorFormat(*src);
        auto s = std::make_shared<CudaTensorStorageDoubleBuffered>(
            src->data_type_, dims, fmt, m_ctx);
        auto t = std::make_shared<CudaTensor>(s, dims, fmt);

        m_flips.push_back(s);
        t->copyFromLocked(*src);
        m_tensors[src] = t;
        addPrePostOp(src, t, s, bt.second);
    }

    for(const auto &bt : bts) {
        if(!(bt.second & BTM::GRADIENT))
            continue;

        auto src = bt.first;
        auto dims = src->dims_.n(m_batch_size);

        auto fmt = tensorFormat(*src);
        auto s = std::make_shared<CudaTensorStorageDoubleBuffered>(
            src->data_type_, dims, fmt, m_ctx);
        auto g = std::make_shared<CudaTensor>(s, dims, fmt);
        m_flips.push_back(s);

        auto t = lower_tensor(src);
        t->m_grad = g;
        addPrePostOp(src, g, s, bt.second);
    }
}

std::shared_ptr<Program>
CudaContext::createProgram(const Graph &g, const ProgramConfig &pc,
                           const BatchedTensors &bts)
{
    std::scoped_lock lock(m_mutex);

    auto p = std::make_shared<CudaProgram>(
        shared_from_this(), pc.tensor_layout, pc.batch_size,
        pc.initial_learning_rate, pc.stop_check, pc.ui);

    p->setupBatchedTensors(bts);

    auto nodes = applyTransforms(CUDA_TRANSFORM_ALL, *p, g.nodes_);

    if(pc.training) {
        auto train_nodes = applyTransforms(CUDA_TRANSFORM_TRAINING, *p, nodes);

        int cnt = 0;
        for(const auto &n : train_nodes) {
            const char *err = find_operation(*n)->setup(*p, *n, true);
            if(err) {
                fprintf(
                    stderr,
                    "Unable to create training operation for %s (#%d)-- %s\n",
                    n->type_.c_str(), cnt, err);
                n->print();
                exit(1);
            }
            cnt++;
        }

        assert(p->m_infer_operations.empty());

        p->m_train_operations.insert(p->m_train_operations.end(),
                                     p->m_fwd_operations.begin(),
                                     p->m_fwd_operations.end());
        p->m_train_operations.insert(p->m_train_operations.end(),
                                     p->m_bwd_operations.begin(),
                                     p->m_bwd_operations.end());
        p->m_train_operations.insert(p->m_train_operations.end(),
                                     p->m_upd_operations.begin(),
                                     p->m_upd_operations.end());

        p->m_fwd_operations.clear();
        p->m_bwd_operations.clear();
        p->m_upd_operations.clear();
    }

    if(pc.inference) {
        auto infer_nodes = applyTransforms(CUDA_TRANSFORM_INFERENCE, *p, nodes);
        for(const auto &n : infer_nodes) {
            const char *err = find_operation(*n)->setup(*p, *n, false);
            if(err) {
                fprintf(stderr,
                        "Unable to create inference operation for %s -- %s\n",
                        n->type_.c_str(), err);
                n->print();
                exit(1);
            }
        }
    }

    for(const auto &kv : p->m_load_map) {
        p->m_load_operations.push_back(kv.second);
    }

    p->m_load_map.clear();
    return p;
}

void
CudaProgram::finalize()
{
    if(m_finalized)
        return;
    m_finalized = true;

    if(m_train_operations.size()) {
        m_train_operations =
            reduceLiveranges(m_train_operations, m_exported_storage);

        m_train_memory_layout =
            memoryLayout(m_train_operations, m_exported_storage);
        m_tensor_mem.request(m_train_memory_layout->size_);
    }

    if(m_infer_operations.size()) {
        m_infer_memory_layout =
            memoryLayout(m_infer_operations, m_exported_storage);
        m_tensor_mem.request(m_infer_memory_layout->size_);
    }

    m_tensor_mem.alloc();

    m_ctx->m_workspace.alloc();

    printf("Tensor memory: %zd MB\n", m_tensor_mem.requested() / 1048576);
    printf("Workspace memory: %zd MB\n",
           m_ctx->m_workspace.requested() / 1048576);
}

void
CudaContext::print() const
{
    size_t memfree = 0, memtotal = 0;
    cudaMemGetInfo(&memfree, &memtotal);
    printf("   Free memory: %zd kbyte\n", memfree / 1024);
    printf("  Total memory: %zd kbyte\n", memtotal / 1024);
}

std::string
CudaContext::info() const
{
    char str[1024];

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, m_deviceId);

    char pciid[32];
    cudaDeviceGetPCIBusId(pciid, sizeof(pciid), m_deviceId);

    snprintf(str, sizeof(str),
             "Device:%s (%d.%d) Concurrent:%s CanMapHostMem:%s TensorCores:%s "
             "id:%d at %s\n",
             prop.name, prop.major, prop.minor,
             prop.concurrentKernels ? "yes" : "no",
             prop.canMapHostMemory ? "yes" : "no",
             m_tensor_cores ? "yes" : "no", m_deviceId, pciid);

    return str;
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
            storage.insert(t->m_storage);
        }
        for(auto const &t : op->getOutputs()) {
            storage.insert(t->m_storage);
        }
    }

    for(auto const &s : storage) {
        int id = s->m_id;
        fprintf(fp, "node [shape=box,label=\"T%d\"]; T%d;\n", id, id);
    }
    fprintf(fp, "\n");

    i = 0;
    for(auto &op : ops) {
        for(auto const &t : op->getInputs()) {
            fprintf(fp, "T%d->O%d;\n", t->m_storage->m_id, i);
        }
        for(auto const &t : op->getOutputs()) {
            fprintf(fp, "O%d->T%d;\n", i, t->m_storage->m_id);
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
    return dumpGraphFromOps(path, m_train_operations);
}

}  // namespace saga
