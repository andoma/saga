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

std::shared_ptr<Tensor>
CudaContext::resolveTensor(std::shared_ptr<Tensor> src)
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
CudaContext::resolveTensorGradient(std::shared_ptr<Tensor> src)
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

cudnnTensorFormat_t
CudaProgram::tensorFormat(Tensor::DataType data_type)
{
    switch(m_pc.tensor_layout) {
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

    auto it = m_ctx->m_tensors.find(src);
    if(it != m_ctx->m_tensors.end()) {
        return it->second;
    }

    Dims dims = src->dims_;

    if(minimum_rank) {
        while(dims.size() < minimum_rank) dims.insert(dims.begin(), 1);
    }

    dims = dims.batch(m_pc.batch_size);

    auto t = std::make_shared<CudaTensor>(
        src->data_type_, dims, tensorFormat(*src), m_ctx, src->name_);

    t->copyFromLocked(*src, 0);
    m_ctx->m_tensors[src] = t;
    return t;
}

std::shared_ptr<CudaTensor>
CudaProgram::lower_tensor(std::shared_ptr<Tensor> src,
                          const CudaTensor &blueprint)
{
    if(src == nullptr)
        return nullptr;

    auto it = m_ctx->m_tensors.find(src);
    if(it != m_ctx->m_tensors.end()) {
        return it->second;
    }

    auto t = std::make_shared<CudaTensor>(src->data_type_, blueprint);
    t->copyFromLocked(*src, 0);
    m_ctx->m_tensors[src] = t;
    return t;
}

std::shared_ptr<CudaTensor>
CudaProgram::lower_tensor(std::shared_ptr<Tensor> src,
                          cudnnTensorFormat_t tensor_format)
{
    if(src == nullptr)
        return nullptr;

    auto it = m_ctx->m_tensors.find(src);
    if(it != m_ctx->m_tensors.end()) {
        return it->second;
    }

    auto t = std::make_shared<CudaTensor>(src->data_type_,
                                          src->dims_.batch(m_pc.batch_size),
                                          tensor_format, m_ctx, src->name_);

    t->copyFromLocked(*src, 0);
    m_ctx->m_tensors[src] = t;
    return t;
}

void
CudaProgram::setupTensorStorage(std::shared_ptr<CudaMemoryLayout> cml)
{
    if(!cml)
        return;

    void *base = m_ctx->m_tensor_mem.ptr();
    for(const auto &kv : cml->table_) {
        void *ptr = (void *)((char *)base + kv.second);
        //    printf("Tensor T%d gets %p\n", kv.first->id_, ptr);
        kv.first->setTmpMem(ptr);
    }
}

void
CudaProgram::detect_anomaly(
    const CudaOperation &op,
    const std::vector<std::shared_ptr<CudaTensor>> &tensors, const char *what)
{
    for(const auto &t : tensors) {
        t->detect_anomaly(&m_aux->anomaly);
        cudaStreamSynchronize(m_ctx->m_stream);
        if(m_aux->anomaly) {
            fprintf(stderr, "\n\n *** Anomaly in tensor T%d (%s)\n",
                    t->storage_id(), what);

            op.dump(stderr, 1);

            for(const auto &t : op.getInputs()) {
                t->print_anomaly("I");
                t->printStats("I");
            }
            for(const auto &t : op.getOutputs()) {
                t->print_anomaly("O");
                t->printStats("O");
            }

            printf("Pre invalidate: ");
            for(auto &t : op.m_pre_invalidate) {
                printf("T%d ", t->m_id);
            }
            printf("\n");

            printf("Post invalidate: ");
            for(auto &t : op.m_post_invalidate) {
                printf("T%d ", t->m_id);
            }
            printf("\n");

            FILE *fp = fopen("anomaly.txt", "w");
            if(fp != NULL) {
                printf("Dumping program to anomaly.txt\n");
                dump(fp, true);
                fclose(fp);
            }
            abort();
        }
    }
}

bool
CudaProgram::runOps(const CudaOps &ops, long batch, bool anomaly_detect)
{
    for(const auto &op : ops) {
        if(anomaly_detect) {
            for(auto &t : op->m_pre_invalidate) {
                t->invalidate();
            }
            detect_anomaly(*op, op->getInputs(), "input");
        }

        const char *err = op->exec(*this, batch);
        if(err) {
            fprintf(stderr, "\nOp %s failed: %s\n", op->name().c_str(), err);
            op->dump(stderr, true);
            return false;
        }

        if(anomaly_detect) {
            detect_anomaly(*op, op->getOutputs(), "output");
            for(auto &t : op->m_post_invalidate) {
                t->invalidate();
            }
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

void
CudaProgram::prep(long batches)
{
    finalize();

    run_batched_tensor_callbacks(m_pc.pre_ops, 0, m_pre_batched_tensors);

    flipDoubleBufferedTensors();

    if(m_pc.ui)
        m_pc.ui->updateBatchInfo(m_pc.batch_size, batches);

    cudaStreamSynchronize(m_ctx->m_stream);

    m_total_samples = 0;
}

ExecResult
CudaProgram::step(long batch, long batches)
{
    setupTensorStorage(m_memory_layout);
    if(!runOps(m_ops, batch, m_pc.anomaly_detect)) {
        return ExecResult::ERROR;
    }

    if(batch > 0) {
        run_batched_tensor_callbacks(m_pc.post_ops, batch - 1,
                                     m_post_batched_tensors);
    }
    if(batch < batches - 1) {
        run_batched_tensor_callbacks(m_pc.pre_ops, batch + 1,
                                     m_pre_batched_tensors);
    }
    flipDoubleBufferedTensors();

    if(m_pc.ui) {
        m_pc.ui->updateProgress(m_index, m_total_samples);
    }

    m_total_samples += m_pc.batch_size;

    cudaStreamSynchronize(m_ctx->m_stream);

    if(m_mp_enabled) {
        if(m_aux->inf || m_aux->nan) {
            m_mp_scaling *= 0.5;
        } else if(m_aux->range > 1) {
            m_mp_scaling *= 0.9;
        } else {
            m_mp_scaling *= 1.02;
        }

        m_aux->range = 0;
        m_aux->inf = 0;
        m_aux->nan = 0;
        if(m_pc.ui) {
            m_pc.ui->updateMpScaling(m_index, m_mp_scaling);
        }
    }
    return ExecResult::OK;
}

void
CudaProgram::post(long batches)
{
    run_batched_tensor_callbacks(m_pc.post_ops, batches - 1,
                                 m_post_batched_tensors);
}

ExecResult
CudaProgram::run(long batches, StopCheck stop_check)
{
    if(batches == 0)
        return ExecResult::OK;

    if(stop_check && stop_check())
        return ExecResult::STOPPED;

    prep(batches);

    for(long i = 0; i < batches; i++) {
        if(m_pc.ui)
            m_pc.ui->updateCurrentBatch(i);

        ExecResult r = step(i, batches);
        if(r != ExecResult::OK)
            return r;

        if(stop_check && stop_check()) {
            return ExecResult::STOPPED;
        }

        if(m_pc.ui)
            updateMemUsage(*m_pc.ui);
    }

    post(batches);
    return ExecResult::OK;
}

ExecResult
CudaContext::multiRun(const std::vector<std::shared_ptr<Program>> &programs,
                      long batches, StopCheck stop_check)
{
    if(programs.size() == 0)
        return ExecResult::OK;

    if(batches == 0)
        return ExecResult::OK;

    if(stop_check && stop_check())
        return ExecResult::STOPPED;

    for(const auto &p : programs) {
        CudaProgram *cp = (CudaProgram *)p.get();
        cp->prep(batches);
    }

    CudaProgram *cp = (CudaProgram *)programs[0].get();
    auto ui = cp->m_pc.ui;

    for(long i = 0; i < batches; i++) {
        if(ui)
            ui->updateCurrentBatch(i);

        for(const auto &p : programs) {
            CudaProgram *cp = (CudaProgram *)p.get();
            ExecResult r = cp->step(i, batches);
            if(r != ExecResult::OK)
                return r;

            if(stop_check && stop_check()) {
                return ExecResult::STOPPED;
            }
        }
        if(ui)
            updateMemUsage(*ui);
    }

    for(const auto &p : programs) {
        CudaProgram *cp = (CudaProgram *)p.get();
        cp->post(batches);
    }
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
CudaOperation::dump(FILE *output, bool full) const
{
    auto inputs = getInputs();
    auto outputs = getOutputs();

    if(full) {
        fprintf(output, "OP: %s %s\n", name().c_str(), info().c_str());
        for(auto const &t : m_pre_invalidate) {
            if(t)
                fprintf(output, "\tE: T%d\n", t->m_id);
        }
        for(auto const &t : inputs) {
            if(t)
                fprintf(output, "\tI: %s\n", t->info().c_str());
        }
        for(auto const &t : outputs) {
            if(t)
                fprintf(output, "\tO: %s\n", t->info().c_str());
        }
        for(auto const &t : m_post_invalidate) {
            if(t)
                fprintf(output, "\tE: T%d\n", t->m_id);
        }
    } else {
        fprintf(output, "%s\n", str().c_str());
    }
}

void
CudaProgram::dump(FILE *output, bool detailed) const
{
    fprintf(output, "\n%zd ops\n", m_ops.size());
    int index = 0;
    for(const auto &op : m_ops) {
        fprintf(output, "#%3d: ", index);
        op->dump(output, detailed);
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
    const char *(*setup)(CudaProgram &p, const Node &n);
};

static std::map<std::string, OpFactory> *cuda_op_factories;

void
CudaRegisterOpFactory(const char *name,
                      const char *(*setup)(CudaProgram &p, const Node &n))
{
    if(!cuda_op_factories)
        cuda_op_factories = new std::map<std::string, OpFactory>;

    (*cuda_op_factories)[name] = OpFactory{.setup = setup};
}

static const char *
no_setup(CudaProgram &p, const Node &n)
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
                          Phase phase)
{
    auto op = CudaBatchAccessOp{high, low, s};

    if(!!(phase & Phase::PRE)) {
        m_pre_batched_tensors.push_back(op);
    }

    if(!!(phase & Phase::POST)) {
        m_ctx->m_exported_storage.push_back(s);
        m_post_batched_tensors.push_back(op);
    }
}

void
CudaProgram::setupBatchedTensors(const BatchedTensors &bts)
{
    // Do values first and gradients in a second pass to make sure lowering
    // is correct

    for(const auto &bt : bts) {
        auto src = bt.first;
        if(src->value())
            continue;  // This was a gradient

        auto dims = src->dims_.batch(m_pc.batch_size);

        auto fmt = tensorFormat(*src);
        auto s = std::make_shared<CudaTensorStorageDoubleBuffered>(
            src->data_type_, dims, fmt, m_ctx);
        auto t = std::make_shared<CudaTensor>(s, dims, fmt);

        m_flips.push_back(s);
        t->copyFromLocked(*src);
        m_ctx->m_tensors[src] = t;
        addPrePostOp(src, t, s, bt.second);
    }

    // Do gradients
    for(const auto &bt : bts) {
        auto src = bt.first;
        auto value = src->value();
        if(!value)
            continue;

        auto dims = src->dims_.batch(m_pc.batch_size);

        auto fmt = tensorFormat(*src);
        auto s = std::make_shared<CudaTensorStorageDoubleBuffered>(
            src->data_type_, dims, fmt, m_ctx);
        auto g = std::make_shared<CudaTensor>(s, dims, fmt);
        m_flips.push_back(s);

        auto t = lower_tensor(value);
        t->m_grad = g;
        addPrePostOp(src, g, s, bt.second);
    }
}

std::shared_ptr<Program>
CudaContext::createProgram(const Graph &g, ProgramType pt,
                           const ProgramConfig &pc,
                           std::optional<std::string> user_name)
{
    auto p = std::make_shared<CudaProgram>(shared_from_this(), pt, pc);

    p->m_name = pt == ProgramType::INFERENCE ? "inference" : "training";

    if(user_name)
        p->m_name = *user_name;

    if(pc.ui)
        pc.ui->updateName(p->m_index, p->m_name);

    p->setupBatchedTensors(pc.batched_tensors);

    auto nodes = applyTransforms(CUDA_TRANSFORM_ALL, *p, g.nodes_);

    nodes =
        applyTransforms(pt == ProgramType::INFERENCE ? CUDA_TRANSFORM_INFERENCE
                                                     : CUDA_TRANSFORM_TRAINING,
                        *p, nodes);

    int cnt = 0;
    int tot = nodes.size();
    for(const auto &n : nodes) {
        printf("\033[KInitializing %s : %d%%\r", p->m_name.c_str(),
               (int)(100.0f * cnt / tot));
        fflush(stdout);

        const char *err = find_operation(*n)->setup(*p, *n);
        if(err) {
            fprintf(stderr,
                    "Unable to create operation for %s "
                    "(#%d)-- %s\n",
                    n->type_.c_str(), cnt, err);
            n->print();
            exit(1);
        }
        cnt++;
    }
    printf("\033[KInitializing %s : Done\n", p->m_name.c_str());

    if(pc.anomaly_detect) {
        // Any tensors written by the bwd operations may contain
        // nonfinite numbers
        for(const auto &op : p->m_bwd_operations) {
            for(auto &t : op->getOutputs()) {
                t->m_storage->m_nonfinite_is_valid = true;
            }
        }
    }

    if(pt == ProgramType::INFERENCE) {
        // For inference type programs, these should be empty
        assert(p->m_bwd_operations.empty());
        assert(p->m_upd_operations.empty());
    }

    return p;
}

void
CudaProgram::finalize()
{
    if(m_finalized) {
        return;
    }
    m_finalized = true;

    m_ops.insert(m_ops.end(), m_fwd_operations.begin(), m_fwd_operations.end());
    m_ops.insert(m_ops.end(), m_bwd_operations.begin(), m_bwd_operations.end());
    m_ops.insert(m_ops.end(), m_upd_operations.begin(), m_upd_operations.end());

    m_fwd_operations.clear();
    m_bwd_operations.clear();
    m_upd_operations.clear();

    if(m_ops.size()) {
        m_ops = reduceLiveranges(m_ops, m_ctx->m_exported_storage);

        m_memory_layout =
            memoryLayout(m_ops, m_ctx->m_exported_storage, m_pc.anomaly_detect);

        m_ctx->m_tensor_mem.request(m_memory_layout->size_);
    }

    m_ctx->m_tensor_mem.alloc();

    m_ctx->m_workspace.alloc();
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
    return dumpGraphFromOps(path, m_ops);
}

}  // namespace saga
