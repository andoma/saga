// -*-c++-*-

#pragma once

#include <memory>
#include <mutex>
#include <map>

#include <cudnn.h>
#include <cublas_v2.h>
#ifdef HAVE_NVIDIA_ML
#include <nvml.h>
#endif

#include "cuda_aux.hpp"

#define chkCUDNN(expression)                                              \
    {                                                                     \
        const cudnnStatus_t cudnn_status__ = (expression);                \
        if(cudnn_status__ != CUDNN_STATUS_SUCCESS) {                      \
            fprintf(stderr, "CUDNN error at %s:%d in %s: %s\n", __FILE__, \
                    __LINE__, __FUNCTION__,                               \
                    cudnnGetErrorString(cudnn_status__));                 \
            abort();                                                      \
        }                                                                 \
    }

#define chkCuda(expression)                                              \
    {                                                                    \
        cudaError_t cuda_status__ = (expression);                        \
        if(cuda_status__) {                                              \
            fprintf(stderr, "CUDA error at %s:%d in %s: %s\n", __FILE__, \
                    __LINE__, __FUNCTION__,                              \
                    cudaGetErrorString(cuda_status__));                  \
            abort();                                                     \
        }                                                                \
    }

namespace saga {

class CudaTensor;
class CudaOperation;
class CudaTensorStorage;
class CudaTensorStorageDoubleBuffered;
struct CudaMemoryLayout;

struct CudaTmpMem {
    CudaTmpMem &operator=(CudaTmpMem const &) = delete;
    CudaTmpMem(CudaTmpMem const &) = delete;

    CudaTmpMem(size_t initial = 0) : m_requested(initial) {}

    ~CudaTmpMem() { chkCuda(cudaFree(m_ptr)); }

    void request(size_t size) { m_requested = std::max(m_requested, size); }

    void alloc()
    {
        if(m_requested <= m_size)
            return;

        chkCuda(cudaFree(m_ptr));

        m_size = m_requested;
        chkCuda(cudaMallocManaged(&m_ptr, m_size, cudaMemAttachGlobal));
    }

    void *ptr() const { return m_ptr; }

    size_t size() const { return m_size; }

    size_t requested() const { return m_requested; }

private:
    void *m_ptr = nullptr;
    size_t m_size = 0;
    size_t m_requested = 0;
};

class CudaContext : public Context,
                    public std::enable_shared_from_this<CudaContext> {
public:
    CudaContext(const std::shared_ptr<UI> &ui, int deviceId)
      : m_ui(ui), m_deviceId(deviceId), m_workspace(4096)
    {
    }

    int init();

    std::shared_ptr<Program> createMultiProgram(
        const std::vector<ProgramSource> &sources, ProgramType pt,
        const ProgramConfig &pc) override;

    std::shared_ptr<Tensor> resolveTensor(std::shared_ptr<Tensor> t) override;

    void reset() override
    {
        m_exported_storage.clear();
        m_tensors.clear();
        m_deferred_copy.clear();
        m_tensor_storage_id_gen = 0;
        m_program_index_generator = 0;
    }

    virtual int getId() override { return m_deviceId; }

    virtual void bindToHostThread() override { cudaSetDevice(m_deviceId); }

    const std::shared_ptr<UI> m_ui;

    const int m_deviceId;

    int m_ui_row;

    CudaTmpMem m_workspace;
    CudaTmpMem m_tensor_mem;

    cudaStream_t m_stream = 0;
    cudnnHandle_t m_cudnn = NULL;
    cublasHandle_t m_cublas = NULL;
#ifdef HAVE_NVIDIA_ML
    nvmlDevice_t m_nvmldev = NULL;
#endif
    int m_num_sm;
    bool m_tensor_cores = false;

    int m_tensor_storage_id_gen = 0;

    int m_program_index_generator = 0;

    // Tensors we've handed out external handles to
    // Their storage need to be kept alive all the time
    std::vector<std::shared_ptr<CudaTensorStorage>> m_exported_storage;

    std::unordered_map<std::shared_ptr<Tensor>, std::shared_ptr<CudaTensor>>
        m_tensors;

    std::unordered_map<std::shared_ptr<CudaTensor>, std::shared_ptr<Tensor>>
        m_deferred_copy;

    std::map<std::string, int> m_algo_hash;

    void updateGpuStats();
};

struct CudaEngine : public Engine {
    CudaEngine(const std::shared_ptr<UI> &ui) : m_ui(ui) {}

    ~CudaEngine() {}

    std::vector<std::shared_ptr<Context>> createContexts(bool multi) override;

    const std::shared_ptr<UI> m_ui;
};

using CudaOp = std::shared_ptr<CudaOperation>;
using CudaOps = std::vector<CudaOp>;

struct CudaProgramUnit {
    Nodes m_transformed;

    int m_batch_size;

    CudaOps m_fwd_operations;
    CudaOps m_bwd_operations;
    CudaOps m_tail_operations;

    void fwd(const std::shared_ptr<CudaOperation> &op);
    void bwd(const std::shared_ptr<CudaOperation> &op);
    void tail(const std::shared_ptr<CudaOperation> &op);
};

class CudaProgram : public Program {
public:
    CudaProgram(std::shared_ptr<CudaContext> ctx, ProgramType pt,
                const ProgramConfig &pc)
      : m_ctx(ctx)
      , m_index(ctx->m_program_index_generator++)
      , m_ui_row(ctx->m_ui->alloc_row())
      , m_pt(pt)
      , m_pc(pc)
    {
        chkCuda(cudaMallocManaged((void **)&m_aux, 4096, cudaMemAttachGlobal));
        chkCuda(cudaMemset(m_aux, 0, 4096));
    }

    ~CudaProgram() { chkCuda(cudaFree(m_aux)); }

    ExecResult run(long batches, StopCheck stop_check,
                   long batch_offset) override;

    void prep(long batches, long batch_offset);
    ExecResult step(long batch, long batches, long batch_offset);
    void post(long batches, long batch_offset);

    void dump(FILE *output, bool detailed) const override;
    void debug(bool) override;

    const std::shared_ptr<CudaContext> m_ctx;
    const int m_index;
    const int m_ui_row;
    const ProgramType m_pt;

    const ProgramConfig m_pc;

    bool m_debug = false;
    bool m_finalized = false;

    std::string m_name;

    CudaOps m_ops;
    CudaOps m_opt;

    std::vector<CudaProgramUnit> m_units;

    std::vector<std::shared_ptr<CudaTensorStorageDoubleBuffered>> m_flips;

    std::shared_ptr<CudaMemoryLayout> m_memory_layout;

    CudaAux *m_aux;
    float m_mp_scaling{1};
    bool m_mp_enabled = false;

    int64_t m_total_samples{0};
    int64_t m_epoch_start{0};

    void finalize();

    std::unordered_map<
        std::shared_ptr<Tensor>,
        std::pair<std::shared_ptr<CudaTensorStorageDoubleBuffered>, Phase>>
        m_batched_tensors;

    void addBatchedTensors(const ProgramSource &ps);

    void run_batched_tensor_callbacks(const TensorBatchCallback &cb, long batch,
                                      Phase phase);

    std::shared_ptr<CudaTensor> resolveTensor_locked(std::shared_ptr<Tensor> t);

    std::shared_ptr<CudaTensor> resolveTensorGradient_locked(
        std::shared_ptr<Tensor> src);

    cudnnTensorFormat_t tensorFormat(Tensor::DataType data_type);

    cudnnTensorFormat_t tensorFormat(const Tensor &t)
    {
        return tensorFormat(t.data_type_);
    }

    std::shared_ptr<CudaTensor> lower_tensor(const CudaProgramUnit &pu,
                                             std::shared_ptr<Tensor> src,
                                             size_t minimum_rank = 0);

    std::shared_ptr<CudaTensor> lower_tensor(std::shared_ptr<Tensor> src,
                                             const CudaTensor &blueprint);

    std::shared_ptr<CudaTensor> lower_tensor(const CudaProgramUnit &pu,
                                             std::shared_ptr<Tensor> src,
                                             cudnnTensorFormat_t format);

    std::shared_ptr<CudaTensor> lower_grad(const CudaProgramUnit &pu,
                                           std::shared_ptr<Tensor> src,
                                           size_t minimum_rank = 0);

    std::shared_ptr<CudaTensor> lower_grad(const CudaProgramUnit &pu,
                                           std::shared_ptr<Tensor> src,
                                           cudnnTensorFormat_t format);

    float upd(const std::shared_ptr<CudaTensor> &weights,
              const std::shared_ptr<CudaTensor> &gradient);

    std::map<std::shared_ptr<CudaTensor>, std::shared_ptr<CudaTensor>>
        m_updates;

    void flipDoubleBufferedTensors();

    void setupTensorStorage(std::shared_ptr<CudaMemoryLayout> cml);

    void detect_anomaly(const CudaOperation &op,
                        const std::vector<std::shared_ptr<CudaTensor>> &tensors,
                        const char *what);

    bool runOps(const CudaOps &ops, long batch, bool anomaly_detect = false);

    bool check_anomaly();

    bool dumpGraphFromOps(const char *path, const CudaOps &ops);

    bool dumpGraph(const char *path) override;

    int getUiRowId() const override { return m_ui_row; }

    ProgramType getType() const override { return m_pt; }

    CudaOps create_optimizers();

    CudaOp create_optimizer(Tensor::DataType dt);
};

class CudaOperation {
public:
    CudaOperation &operator=(CudaOperation const &) = delete;
    CudaOperation(CudaOperation const &) = delete;

    virtual ~CudaOperation() {}
    virtual const char *exec(CudaProgram &p, long batch) = 0;

    virtual std::vector<std::shared_ptr<CudaTensor>> getInputs() const = 0;
    virtual std::vector<std::shared_ptr<CudaTensor>> getOutputs() const = 0;
    virtual bool killOutput(std::shared_ptr<CudaTensorStorage> t)
    {
        return false;
    }

    virtual std::shared_ptr<CudaOperation> getSyncOp() { return nullptr; };

    void dump(FILE *output, bool full = false) const;

    std::string name() const { return m_kind; }

    virtual std::string info() const { return ""; };

    std::string str() const;

    std::vector<std::shared_ptr<CudaTensorStorage>> m_pre_invalidate;
    std::vector<std::shared_ptr<CudaTensorStorage>> m_post_invalidate;

protected:
    CudaOperation(std::string kind) : m_kind(kind) {}

    CudaOperation() = delete;

    std::string m_kind;
};

#define CPPGLUE(a, b) a##b
#define CPPJOIN(a, b) CPPGLUE(a, b)

void CudaRegisterOpFactory(const char *name,
                           const char *(*setup)(CudaProgram &p,
                                                CudaProgramUnit &pu,
                                                const Node &n));

#define REGISTER_CUDA_OP(name, setup)                                      \
    static void __attribute__((constructor)) CPPJOIN(init, __LINE__)(void) \
    {                                                                      \
        CudaRegisterOpFactory(name, setup);                                \
    }

void CudaRegisterTransform(ProgramType type,
                           Nodes (*op)(CudaProgram &p, CudaProgramUnit &pu,
                                       const Nodes &nodes));

#define REGISTER_CUDA_TRANSFORM(prio, type, op)           \
    static void __attribute__((constructor(1000 + prio))) \
    CPPJOIN(init, __LINE__)(void)                         \
    {                                                     \
        CudaRegisterTransform(type, op);                  \
    }
}  // namespace saga
