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
class CudaMemoryLayout;

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

private:
    void *m_ptr = nullptr;
    size_t m_size = 0;
    size_t m_requested = 0;
};

class CudaContext : public Context,
                    public std::enable_shared_from_this<CudaContext> {
public:
    CudaContext() : m_workspace(4096) {}

    int init();

    std::shared_ptr<Program> createProgram(
        const Graph &g, const ProgramConfig &pc,
        const BatchTensorAccessors &accessors);

    void print() override;

    CudaTmpMem m_workspace;

    cudaStream_t m_stream = 0;
    cudnnHandle_t m_cudnn = NULL;
    cublasHandle_t m_cublas = NULL;
#ifdef HAVE_NVIDIA_ML
    nvmlDevice_t m_nvmldev = NULL;
#endif
    int m_deviceId;
    std::mutex m_mutex;

    int m_tensor_storage_id_gen = 0;

    bool m_tensor_cores = false;
};

struct CudaBatchAccessOp {
    std::shared_ptr<CudaTensor> m_tensor;
    std::shared_ptr<CudaTensorStorageDoubleBuffered> m_storage;
    BatchTensorAccessFn m_fn;
    bool m_prefetch = false;
};

typedef std::vector<CudaBatchAccessOp> CudaBatchAccessOps;

typedef std::vector<std::shared_ptr<CudaOperation>> CudaOps;

class CudaProgram : public Program {
public:
    CudaProgram(std::shared_ptr<CudaContext> ctx, TensorLayout tensor_layout,
                int batch_size, float learning_rate, StopCheck stop_check,
                bool print_progress)
      : m_ctx(ctx)
      , m_tensor_layout(tensor_layout)
      , m_batch_size(batch_size)
      , m_learning_rate(learning_rate)
      , m_mp_scaling(batch_size)
      , m_stop_check(stop_check)
      , m_print_progress(print_progress)
    {
        chkCuda(cudaMallocManaged(&m_check_result, sizeof(int),
                                  cudaMemAttachGlobal));
        chkCuda(cudaMemset(m_check_result, 0, sizeof(int)));
    }

    ~CudaProgram() { chkCuda(cudaFree(m_check_result)); }

    std::shared_ptr<Tensor> resolveTensor(std::shared_ptr<Tensor> t) override;
    std::shared_ptr<Tensor> resolveTensorGradient(
        std::shared_ptr<Tensor> src) override;
    ExecResult infer(long batches) override;
    ExecResult train(long batches) override;
    void print(bool detailed) const override;
    void debug(bool) override;

    const std::shared_ptr<CudaContext> m_ctx;
    const TensorLayout m_tensor_layout;
    const int m_batch_size;
    const float m_learning_rate;
    bool m_debug = false;

    bool m_finalized = false;

    // Tensors we've handed out external handles to
    // Their storage need to be kept alive all the time
    std::vector<std::shared_ptr<CudaTensorStorage>> m_exported_storage;

    std::unordered_map<std::shared_ptr<Tensor>, std::shared_ptr<CudaTensor>>
        m_tensors;

    std::unordered_map<std::shared_ptr<Tensor>, std::shared_ptr<CudaOperation>>
        m_load_map;

    CudaOps m_load_operations;
    CudaOps m_infer_operations;
    CudaOps m_train_operations;

    CudaOps m_fwd_operations;
    CudaOps m_bwd_operations;
    CudaOps m_upd_operations;

    CudaBatchAccessOps m_infer_pre;
    CudaBatchAccessOps m_infer_post;
    CudaBatchAccessOps m_train_pre;
    CudaBatchAccessOps m_train_post;

    std::vector<std::shared_ptr<CudaTensorStorageDoubleBuffered>> m_flips;

    std::shared_ptr<CudaMemoryLayout> m_train_memory_layout;
    std::shared_ptr<CudaMemoryLayout> m_infer_memory_layout;

    CudaTmpMem m_tensor_mem;

    void *m_check_result;
    float m_mp_scaling;
    bool m_mp_enabled = false;

    StopCheck m_stop_check;
    bool m_print_progress = true;
    bool m_print_progress_pending_nl = false;
    time_t m_print_progress_ts = 0;

    std::map<std::string, int> m_algo_hash;

    void finalize();

    std::shared_ptr<CudaTensor> resolveTensor_locked(std::shared_ptr<Tensor> t);

    std::shared_ptr<CudaTensor> resolveTensorGradient_locked(
        std::shared_ptr<Tensor> src);

    cudnnTensorFormat_t tensorFormat(Tensor::DataType data_type);

    std::shared_ptr<CudaTensor> lower_tensor(std::shared_ptr<Tensor> src,
                                             size_t rank = 0);

    std::shared_ptr<CudaTensor> lower_tensor_batch(std::shared_ptr<Tensor> src,
                                                   cudnnTensorFormat_t format);

    std::shared_ptr<CudaTensor> lower_tensor_batch(std::shared_ptr<Tensor> src);

    std::shared_ptr<CudaTensor> lower_tensor_batch(std::shared_ptr<Tensor> src,
                                                   const CudaTensor &blueprint);

    void infer(const std::shared_ptr<CudaOperation> &op);

    void fwd(const std::shared_ptr<CudaOperation> &op);
    void bwd(const std::shared_ptr<CudaOperation> &op);
    void upd(const std::shared_ptr<CudaOperation> &op);

    void setupAccessors(const BatchTensorAccessors &accessors);

    void addPrePostOp(std::shared_ptr<CudaTensor> t,
                      std::shared_ptr<CudaTensorStorageDoubleBuffered> s,
                      const BatchTensorAccess &a);

    void issueBatchAccessOps(const CudaBatchAccessOps ops, long batch);

    void flipDoubleBufferedTensors();

    void setupTensorStorage(std::shared_ptr<CudaMemoryLayout> cml);

    bool runOps(const CudaOps &ops, long batch);

    void progress(const char *what, long i, long batches, float mp_scaling,
                  int64_t start_time);

    void progressDone();

    bool dumpGraphFromOps(const char *path, const CudaOps &ops);

    bool dumpGraph(const char *path) override;
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

    void print(bool full = false) const;

    std::string name() const { return m_kind; }

    virtual std::string info() const { return ""; };

    std::string str() const;

protected:
    CudaOperation(std::string kind) : m_kind(kind) {}

    CudaOperation() = delete;

    std::string m_kind;
};

#define CPPGLUE(a, b) a##b
#define CPPJOIN(a, b) CPPGLUE(a, b)

void CudaRegisterOpFactory(const char *name,
                           const char *(*setup)(CudaProgram &p, const Node &n,
                                                bool training));

#define REGISTER_CUDA_OP(name, setup)                                      \
    static void __attribute__((constructor)) CPPJOIN(init, __LINE__)(void) \
    {                                                                      \
        CudaRegisterOpFactory(name, setup);                                \
    }

enum CudaTransformType {
    CUDA_TRANSFORM_ALL,
    CUDA_TRANSFORM_TRAINING,
    CUDA_TRANSFORM_INFERENCE,
};

void CudaRegisterTransform(CudaTransformType type,
                           Nodes (*op)(CudaProgram &p, const Nodes &nodes));

#define REGISTER_CUDA_TRANSFORM(prio, type, op)           \
    static void __attribute__((constructor(1000 + prio))) \
    CPPJOIN(init, __LINE__)(void)                         \
    {                                                     \
        CudaRegisterTransform(type, op);                  \
    }
}  // namespace saga
