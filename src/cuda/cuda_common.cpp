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
#include "engine.hpp"

#include "cuda_common.hpp"
#include "cuda_tensor.hpp"
#include "cuda_analysis.hpp"

namespace saga {

void
CudaContext::updateGpuStats()
{
    size_t memfree = 0, memtotal = 0;
    cudaMemGetInfo(&memfree, &memtotal);
    size_t used = memtotal - memfree;

    m_ui->updateCell(m_ui_page, 0, 2, UI::Align::RIGHT, "Mem:");
    m_ui->updateCell(m_ui_page, 0, 3, UI::Align::LEFT, "%zd / %zd", used >> 20,
                     memtotal >> 20);
}

int
CudaContext::init()
{
#ifdef HAVE_NVIDIA_ML
    nvmlInit();
#endif
    cudaSetDevice(m_deviceId);

    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, m_deviceId);

    m_ui->updateCell(m_ui_page, 0, 0, UI::Align::LEFT, "%s", prop.name);
    m_ui->updateCell(m_ui_page, 0, 1, UI::Align::LEFT, "Dev #%d", m_deviceId);

    chkCuda(cudaStreamCreateWithFlags(&m_stream, 0));
    char pciid[32];
    cudaDeviceGetPCIBusId(pciid, sizeof(pciid), m_deviceId);

    cudaDeviceGetAttribute(&m_num_sm, cudaDevAttrMultiProcessorCount,
                           m_deviceId);

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

std::vector<std::shared_ptr<Context>>
CudaEngine::createContexts(bool multi)
{
    std::vector<std::shared_ptr<Context>> ret;

    int num_devices;
    cudaGetDeviceCount(&num_devices);

    int num_contexts = num_devices;

    const char *e = getenv("SAGA_CUDA_CONTEXTS");
    if(e)
        num_contexts = atoi(e);

    m_nodes = 0;

    for(int i = 0; i < num_contexts; i++) {
        auto ctx = std::make_shared<CudaContext>(m_ui, m_nodes,
                                                 std::min(i, num_devices - 1));
        ctx->init();
        ctx->m_engine = shared_from_this();
        ret.push_back(ctx);
        m_nodes++;
        if(!multi)
            break;
    }

#ifdef HAVE_NCCL
    if(m_nodes > 1) {
        m_nccl_comms.resize(m_nodes);
        ncclResult_t r = ncclCommInitAll(m_nccl_comms.data(), m_nodes, NULL);

        if(r != ncclSuccess) {
            fprintf(stderr, "Unable to init NCCL -- %s\n",
                    ncclGetErrorString(r));
            exit(1);
        }
    }
#endif
    return ret;
}

static std::shared_ptr<Engine>
createCudaEngine(const std::shared_ptr<UI> &ui)
{
    assert(ui.get());
    return std::make_shared<CudaEngine>(ui);
}

static void __attribute__((constructor)) registerCudaEngine(void)
{
    if(getenv("SAGA_DISABLE_CUDA"))
        return;
    registerEngineFactory("cuda", &createCudaEngine);
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

CudaOpArgs
CudaOperation::getInputs()
{
    auto list = listInputs();

    for(auto &t : listOutputs()) {
        float *betap = getBeta(t.second);
        if(betap && *betap) {
            list.push_back(t);
        }
    }
    return list;
}

std::string
CudaOperation::str()
{
    std::stringstream ss;
    const char *sep = "";

    for(auto const &t : getOutputs()) {
        float *betap = getBeta(t.second);
        const char *mode = betap && *betap ? "+" : "";
        ss << sep << mode << t.first << ":" << t.second->shortname();
        sep = ", ";
    }

    ss << " = " << name() << "(";
    sep = "";
    for(auto const &t : getInputs()) {
        ss << sep << t.first << ":" << t.second->shortname();
        sep = ", ";
    }
    auto inf = info();
    if(inf.size())
        ss << ", " << inf;

    ss << ")";
    return ss.str();
}

void
CudaOperation::dump(FILE *output, bool full)
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
            fprintf(output, "\tI: %3s: %s\n", t.first,
                    t.second->info().c_str());
        }
        for(auto const &t : outputs) {
            fprintf(output, "\tO: %3s: %s\n", t.first,
                    t.second->info().c_str());
        }
        for(auto const &t : m_post_invalidate) {
            if(t)
                fprintf(output, "\tE: %s\n", t->name().c_str());
        }
    } else {
        fprintf(output, "%s\n", str().c_str());
    }
}

//------------------------------------------------------------------------

static std::map<std::string, OpSetup> cuda_op_factories;

void
CudaRegisterOpFactory(const char *name, OpSetup setup)
{
    cuda_op_factories[name] = setup;
}

static OpSetup
find_operation(const Node &n)
{
    auto &p = cuda_op_factories[n.m_type];
    if(!p) {
        return [](CudaProgram &p, CudaProgramUnit &pu, const Node &n) {
            return "operation does not exist";
        };
    }
    return p;
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

        printf("%s:\n", n->m_type.c_str());

        for(const auto &t : n->m_inputs) {
            auto l = p.resolveTensor_locked(t.second);
            printf("\t Input: %s: %s\n", t.first.c_str(),
                   l ? l->info().c_str() : t.second->info().c_str());
        }

        for(const auto &t : n->m_outputs) {
            auto l = p.resolveTensor_locked(t.second);
            printf("\tOutput: %s: %s\n", t.first.c_str(),
                   l ? l->info().c_str() : t.second->info().c_str());
        }

        for(const auto &a : n->m_attributes) {
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

std::shared_ptr<Program>
CudaContext::createMultiProgram(const std::vector<ProgramSource> &sources,
                                ProgramType pt, const ProgramConfig &pc)
{
    cudaSetDevice(m_deviceId);

    auto p = std::make_shared<CudaProgram>(shared_from_this(), pt, pc);

    p->m_name = pt == ProgramType::INFERENCE ? "Inference" : "Training";

    m_ui->updateCell(m_ui_page, p->m_ui_row, 0, UI::Align::LEFT, "%s",
                     p->m_name.c_str());
    m_ui->updateCell(m_ui_page, p->m_ui_row, 4, UI::Align::RIGHT, "Batch:");
    m_ui->updateCell(m_ui_page, p->m_ui_row, 6, UI::Align::RIGHT, "Sample:");

    size_t total_nodes = 0;

    float mp_scaling = 0.0f;

    for(const auto &s : sources) {
        p->addBatchedTensors(s);

        p->m_units.push_back({});

        auto &pu = p->m_units[p->m_units.size() - 1];
        pu.m_batch_size = s.batch_size;
        mp_scaling += s.batch_size;
        pu.m_transformed = applyTransforms(*p, pu, s.graph.m_nodes);
        total_nodes += pu.m_transformed.size();
    }

    p->m_mp_scaling = mp_scaling;

    size_t cnt = 0;
    for(auto &pu : p->m_units) {
        for(const auto &n : pu.m_transformed) {
            m_ui->updateCell(m_ui_page, p->m_ui_row, 1, UI::Align::LEFT,
                             "Init:%d%%", (int)(100.0f * cnt / total_nodes));

            auto op = find_operation(*n);
            try {
                op(*p, pu, *n);
            } catch(const std::exception &e) {
                throw std::runtime_error(
                    fmt("Unable to create operation %s (#%zd) -- %s",
                        n->m_type.c_str(), cnt, e.what()));
            }
            cnt++;
        }

        if(pc.anomaly_detect) {
            // Any tensors written by the bwd operations may contain
            // nonfinite numbers
            for(const auto &op : pu.m_bwd_operations) {
                for(auto &t : op->getOutputs()) {
                    t.second->m_storage->m_nonfinite_is_valid = true;
                }
            }
        }

        if(pt == ProgramType::INFERENCE) {
            // For inference type programs, these should be empty
            assert(pu.m_bwd_operations.empty());
        }
    }

    m_ui->updateCell(m_ui_page, p->m_ui_row, 1, UI::Align::LEFT, "Paused");

    if(pt == ProgramType::INFERENCE) {
        // For inference type programs, these should be empty
        assert(p->m_updates.empty());
    } else if(pt == ProgramType::TRAINING) {
        p->m_opt = p->create_optimizers();
        p->m_updates.clear();
    }

    return p;
}

#if 0
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
#endif

}  // namespace saga
