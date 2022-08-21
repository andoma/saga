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
CudaProgramUnit::fwd(const std::shared_ptr<CudaOperation> &op)
{
    m_fwd_operations.push_back(op);
}

void
CudaProgramUnit::bwd(const std::shared_ptr<CudaOperation> &op)
{
    m_bwd_operations.insert(m_bwd_operations.begin(), op);
}

void
CudaProgramUnit::tail(const std::shared_ptr<CudaOperation> &op)
{
    m_tail_operations.push_back(op);
}

float
CudaProgram::upd(const std::shared_ptr<CudaTensor> &weights,
                 const std::shared_ptr<CudaTensor> &gradient)
{
    auto &p = m_updates[weights];
    if(p) {
        assert(p.get() == gradient.get());
        return 1.0f;
    } else {
        p = gradient;
        return 0;
    }
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
CudaProgram::lower_tensor(const CudaProgramUnit &pu,
                          std::shared_ptr<Tensor> src, size_t minimum_rank)
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

    dims = dims.batch(pu.m_batch_size);

    auto t = std::make_shared<CudaTensor>(
        src->data_type_, dims, tensorFormat(*src), m_ctx, src->name_);

    m_ctx->m_deferred_copy[t] = src;
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
    m_ctx->m_deferred_copy[t] = src;
    m_ctx->m_tensors[src] = t;
    return t;
}

std::shared_ptr<CudaTensor>
CudaProgram::lower_tensor(const CudaProgramUnit &pu,
                          std::shared_ptr<Tensor> src,
                          cudnnTensorFormat_t tensor_format)
{
    if(src == nullptr)
        return nullptr;

    auto it = m_ctx->m_tensors.find(src);
    if(it != m_ctx->m_tensors.end()) {
        return it->second;
    }

    auto t = std::make_shared<CudaTensor>(src->data_type_,
                                          src->dims_.batch(pu.m_batch_size),
                                          tensor_format, m_ctx, src->name_);

    m_ctx->m_deferred_copy[t] = src;
    m_ctx->m_tensors[src] = t;
    return t;
}

std::shared_ptr<CudaTensor>
CudaProgram::lower_grad(const CudaProgramUnit &pu, std::shared_ptr<Tensor> src,
                        size_t minimum_rank)
{
    if(src == nullptr)
        return nullptr;
    src = src->grad();
    assert(src.get());
    return lower_tensor(pu, src, minimum_rank);
}

std::shared_ptr<CudaTensor>
CudaProgram::lower_grad(const CudaProgramUnit &pu, std::shared_ptr<Tensor> src,
                        cudnnTensorFormat_t format)
{
    if(src == nullptr)
        return nullptr;
    src = src->grad();
    assert(src.get());
    return lower_tensor(pu, src, format);
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
            fprintf(stderr, "\n\n *** Anomaly in tensor %s (%s)\n",
                    t->storage_name().c_str(), what);

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
                for(const auto &id : t->m_idmap) {
                    printf("T%d ", id.second);
                }
            }
            printf("\n");

            printf("Post invalidate: ");
            for(auto &t : op.m_post_invalidate) {
                for(const auto &id : t->m_idmap) {
                    printf("T%d ", id.second);
                }
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
updateGpuStats(UI &ui)
{
    size_t memfree = 0, memtotal = 0;
    cudaMemGetInfo(&memfree, &memtotal);
    size_t used = memtotal - memfree;

    ui.updateCell(0, 2, UI::Align::RIGHT, "GPU-Mem:");
    ui.updateCell(0, 3, UI::Align::LEFT, "%zd / %zd", used >> 20,
                  memtotal >> 20);
}

void
CudaProgram::prep(long batches)
{
    finalize();

    if(m_pc.ui) {
        m_pc.ui->updateCell(m_ui_row, 1, UI::Align::LEFT, "Run");
    }

    run_batched_tensor_callbacks(m_pc.pre_ops, 0, Phase::PRE);

    flipDoubleBufferedTensors();

    cudaStreamSynchronize(m_ctx->m_stream);

    m_total_samples = 0;
}

ExecResult
CudaProgram::step(long batch, long batches)
{
    if(m_pc.ui) {
        m_pc.ui->updateCell(m_ui_row, 5, UI::Align::LEFT, "%ld / %ld", batch,
                            batches);
    }

    setupTensorStorage(m_memory_layout);
    if(!runOps(m_ops, batch, m_pc.anomaly_detect)) {
        return ExecResult::ERROR;
    }

    if(batch > 0) {
        run_batched_tensor_callbacks(m_pc.post_ops, batch - 1, Phase::POST);
    }

    if(batch < batches - 1) {
        run_batched_tensor_callbacks(m_pc.pre_ops, batch + 1, Phase::PRE);
    }

    flipDoubleBufferedTensors();
    if(m_pc.ui) {
        m_pc.ui->updateCell(m_ui_row, 7, UI::Align::LEFT, "%d",
                            m_total_samples);
    }

    for(const auto &u : m_units) {
        m_total_samples += u.m_batch_size;
    }

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
            m_pc.ui->updateCell(m_ui_row, 9, UI::Align::LEFT, "%.1e",
                                m_mp_scaling);
        }
    }
    return ExecResult::OK;
}

void
CudaProgram::post(long batches)
{
    if(m_pc.ui) {
        m_pc.ui->updateCell(m_ui_row, 1, UI::Align::LEFT, "Pause");
    }
    run_batched_tensor_callbacks(m_pc.post_ops, batches - 1, Phase::POST);
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
        ExecResult r = step(i, batches);
        if(r != ExecResult::OK)
            return r;

        if(stop_check && stop_check()) {
            return ExecResult::STOPPED;
        }

        if(m_pc.ui)
            updateGpuStats(*m_pc.ui);
    }

    post(batches);
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
                fprintf(output, "\tE: %s\n", t->name().c_str());
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
                fprintf(output, "\tE: %s\n", t->name().c_str());
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
    const char *(*setup)(CudaProgram &p, CudaProgramUnit &pu, const Node &n);
};

static std::map<std::string, OpFactory> *cuda_op_factories;

void
CudaRegisterOpFactory(const char *name,
                      const char *(*setup)(CudaProgram &p, CudaProgramUnit &pu,
                                           const Node &n))
{
    if(!cuda_op_factories)
        cuda_op_factories = new std::map<std::string, OpFactory>;

    (*cuda_op_factories)[name] = OpFactory{.setup = setup};
}

static const char *
no_setup(CudaProgram &p, CudaProgramUnit &pu, const Node &n)
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
    ProgramType type;
    Nodes (*op)(CudaProgram &p, CudaProgramUnit &pu, const Nodes &nodes);
};

static std::vector<CudaNodeTransform> *transforms;

void
CudaRegisterTransform(ProgramType type,
                      Nodes (*op)(CudaProgram &p, CudaProgramUnit &pu,
                                  const Nodes &nodes))
{
    if(!transforms)
        transforms = new std::vector<CudaNodeTransform>;
    transforms->push_back(CudaNodeTransform{.type = type, .op = op});
}

static Nodes
applyTransforms(CudaProgram &p, CudaProgramUnit &pu, const Nodes &nodes)
{
    auto copy = nodes;
    for(auto const &cnt : *transforms) {
        if(!!(p.m_pt & cnt.type))
            copy = cnt.op(p, pu, copy);
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
compute_dx_beta(CudaProgram &p, CudaProgramUnit &pu, const Nodes &nodes)
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
REGISTER_CUDA_TRANSFORM(1000, ProgramType::TRAINING, compute_dx_beta);

void
CudaProgram::addBatchedTensors(const ProgramSource &ps)
{
    for(const auto &bt : ps.batched_tensors) {
        auto src = bt.first;

        auto &p = m_batched_tensors[src];

        if(!p.first) {
            auto dims = src->dims_.batch(ps.batch_size);
            auto fmt = tensorFormat(*src);
            auto s = std::make_shared<CudaTensorStorageDoubleBuffered>(
                src->data_type_, dims, fmt, m_ctx);
            auto t = std::make_shared<CudaTensor>(s, dims, fmt);

            m_flips.push_back(s);
            t->copyFromLocked(*src);
            m_ctx->m_tensors[src] = t;
            p.first = s;
        }

        if(!(p.second & Phase::POST) && !!(bt.second & Phase::POST)) {
            m_ctx->m_exported_storage.push_back(p.first);
        }
        p.second = p.second | bt.second;
    }
}

std::shared_ptr<Program>
CudaContext::createMultiProgram(const std::vector<ProgramSource> &sources,
                                ProgramType pt, const ProgramConfig &pc)
{
    auto p = std::make_shared<CudaProgram>(shared_from_this(), pt, pc);

    p->m_name = pt == ProgramType::INFERENCE ? "Inference" : "Training";

    if(pc.ui) {
        pc.ui->updateCell(p->m_ui_row, 0, UI::Align::LEFT, "%s",
                          p->m_name.c_str());
        pc.ui->updateCell(p->m_ui_row, 4, UI::Align::RIGHT, "Batch:");
        pc.ui->updateCell(p->m_ui_row, 6, UI::Align::RIGHT, "Sample:");
    }

    size_t total_nodes = 0;
    for(const auto &s : sources) {
        p->addBatchedTensors(s);

        p->m_units.push_back({});

        auto &pu = p->m_units[p->m_units.size() - 1];
        pu.m_batch_size = s.batch_size;
        pu.m_transformed = applyTransforms(*p, pu, s.graph.nodes_);
        total_nodes += pu.m_transformed.size();
    }

    size_t cnt = 0;
    for(auto &pu : p->m_units) {
        for(const auto &n : pu.m_transformed) {
            if(pc.ui) {
                pc.ui->updateCell(p->m_ui_row, 1, UI::Align::LEFT, "%d%%",
                                  (int)(100.0f * cnt / total_nodes));
            }

            const char *err = find_operation(*n)->setup(*p, pu, *n);
            if(err) {
                fprintf(stderr,
                        "Unable to create operation for %s "
                        "(#%zd)-- %s\n",
                        n->type_.c_str(), cnt, err);
                n->print();
                exit(1);
            }
            cnt++;
        }

        if(pc.anomaly_detect) {
            // Any tensors written by the bwd operations may contain
            // nonfinite numbers
            for(const auto &op : pu.m_bwd_operations) {
                for(auto &t : op->getOutputs()) {
                    t->m_storage->m_nonfinite_is_valid = true;
                }
            }
        }

        if(pt == ProgramType::INFERENCE) {
            // For inference type programs, these should be empty
            assert(pu.m_bwd_operations.empty());
        }
    }

    if(pc.ui) {
        pc.ui->updateCell(p->m_ui_row, 1, UI::Align::LEFT, "");
    }

    if(pt == ProgramType::INFERENCE) {
        // For inference type programs, these should be empty
        assert(p->m_updates.empty());
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

    for(const auto &[to, from] : m_ctx->m_deferred_copy) {
        to->copyFromLocked(*from, 0);
    }
    m_ctx->m_deferred_copy.clear();

    for(auto &pu : m_units) {
        m_ops.insert(m_ops.end(), pu.m_fwd_operations.begin(),
                     pu.m_fwd_operations.end());
        m_ops.insert(m_ops.end(), pu.m_bwd_operations.begin(),
                     pu.m_bwd_operations.end());

        pu.m_fwd_operations.clear();
        pu.m_bwd_operations.clear();
    }

    for(auto &it : m_updates) {
        m_ops.push_back(optimize(it.first, it.second));
    }

    m_updates.clear();

    if(m_ops.size()) {
        m_ops = reduceLiveranges(m_ops, m_ctx->m_exported_storage);

        m_memory_layout =
            memoryLayout(m_ops, m_ctx->m_exported_storage, m_pc.anomaly_detect);

        m_ctx->m_tensor_mem.request(m_memory_layout->size_);
    }

    m_ctx->m_tensor_mem.alloc();

    m_ctx->m_workspace.alloc();

    if(m_mp_enabled && m_pc.ui) {
        m_pc.ui->updateCell(m_ui_row, 8, UI::Align::RIGHT, "MPS:");
    }
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
        const auto name = s->name();
        fprintf(fp, "node [shape=box,label=\"%s\"]; %s;\n", name.c_str(),
                name.c_str());
    }
    fprintf(fp, "\n");

    i = 0;
    for(auto &op : ops) {
        for(auto const &t : op->getInputs()) {
            const auto name = t->storage_name();
            fprintf(fp, "%s->O%d;\n", name.c_str(), i);
        }
        for(auto const &t : op->getOutputs()) {
            const auto name = t->storage_name();
            fprintf(fp, "O%d->%s;\n", i, name.c_str());
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
