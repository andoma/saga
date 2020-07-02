// -*-c++-*-

#pragma once

#include <memory>
#include <mutex>

#include <cudnn.h>
#include <cublas_v2.h>
#include <nvml.h>


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
  cudaError_t cuda_status__ = (expression);                             \
  if(cuda_status__) {                                                   \
    fprintf(stderr, "CUDA error at %s:%d in %s: %s\n",                  \
            __FILE__, __LINE__, __FUNCTION__,                           \
            cudaGetErrorString(cuda_status__));                         \
    abort();                                                            \
  }                                                                     \
}


namespace saga {

class CudaTensor;
class CudaOperation;
class CudaTensorStorage;
class CudaTensorStorageDoubleBuffered;
class CudaMemoryLayout;


class CudaContext : public Context,
                    public std::enable_shared_from_this<CudaContext> {
public:

  int init();

  std::shared_ptr<Program> createProgram(const Graph &g,
                                         const ProgramConfig &pc,
                                         const BatchTensorAccessors &accessors);

  void print() override;

  cudaStream_t stream_ = 0;
  cudnnHandle_t cudnn_ = NULL;
  cublasHandle_t cublas_ = NULL;
  nvmlDevice_t nvmldev_ = NULL;

  int deviceId_;
  std::mutex mutex_;

  int tensor_storage_id_gen_ = 0;
};


struct CudaBatchAccessOp {
  std::shared_ptr<CudaTensor> tensor_;
  std::shared_ptr<CudaTensorStorageDoubleBuffered> storage_;
  BatchTensorAccessFn fn_;
  bool prefetch_ = false;
};

typedef std::vector<CudaBatchAccessOp> CudaBatchAccessOps;

typedef std::vector<std::shared_ptr<CudaOperation>> CudaOps;

struct CudaTmpMem {

  ~CudaTmpMem()
  {
    chkCuda(cudaFree(ptr_));
  }

  void request(size_t size)
  {
    requested_ = std::max(requested_, size);
  }

  void alloc()
  {
    if(requested_ <= size_)
      return;

    chkCuda(cudaFree(ptr_));

    size_ = requested_;
    chkCuda(cudaMalloc(&ptr_, size_));
  }

  void allocManaged()
  {
    if(requested_ <= size_)
      return;

    chkCuda(cudaFree(ptr_));

    size_ = requested_;
    chkCuda(cudaMallocManaged(&ptr_, size_, cudaMemAttachGlobal));
  }


  void *ptr() const
  {
    return ptr_;
  }

  size_t size() const
  {
    return size_;
  }

private:
  void *ptr_ = nullptr;
  size_t size_ = 0;
  size_t requested_ = 0;
};




class CudaProgram : public Program {
public:

  CudaProgram(std::shared_ptr<CudaContext> ctx,
              TensorLayout tensor_layout,
              int batch_size,
              float learning_rate,
              StopCheck stop_check,
              bool print_progress)
    : ctx_(ctx)
    , tensor_layout_(tensor_layout)
    , batch_size_(batch_size)
    , learning_rate_(learning_rate)
    , mp_scaling_(batch_size)
    , stop_check_(stop_check)
    , print_progress_(print_progress)
  {
    chkCuda(cudaMallocManaged(&check_result_, sizeof(int), cudaMemAttachGlobal));
    chkCuda(cudaMemset(check_result_, 0, sizeof(int)));
  }

  ~CudaProgram()
  {
    chkCuda(cudaFree(check_result_));
  }


  std::shared_ptr<Tensor> resolveTensor(std::shared_ptr<Tensor> t) override;
  std::shared_ptr<Tensor> resolveTensorGradient(std::shared_ptr<Tensor> src) override;
  ExecResult infer(long batches) override;
  ExecResult train(long batches) override;
  void print(bool detailed) const override;
  void debug(bool) override;

  const std::shared_ptr<CudaContext> ctx_;
  const TensorLayout tensor_layout_;
  const int batch_size_;
  const float learning_rate_;
  bool debug_ = false;

  bool finalized_ = false;

  // Tensors we've handed out external handles to
  // Their storage need to be kept alive all the time
  std::vector<std::shared_ptr<CudaTensorStorage>> exported_storage_;

  std::unordered_map<std::shared_ptr<Tensor>,
                     std::shared_ptr<CudaTensor>> tensors_;

  std::unordered_map<std::shared_ptr<Tensor>,
                     std::shared_ptr<CudaOperation>> load_map_;

  CudaOps load_operations_;
  CudaOps infer_operations_;
  CudaOps train_operations_;

  CudaOps fwd_operations_;
  CudaOps bwd_operations_;
  CudaOps upd_operations_;

  CudaBatchAccessOps infer_pre_;
  CudaBatchAccessOps infer_post_;
  CudaBatchAccessOps train_pre_;
  CudaBatchAccessOps train_post_;

  std::vector<std::shared_ptr<CudaTensorStorageDoubleBuffered>> flips_;

  std::shared_ptr<CudaMemoryLayout> train_memory_layout_;
  std::shared_ptr<CudaMemoryLayout> infer_memory_layout_;

  CudaTmpMem workspace_;
  CudaTmpMem tensor_mem_;

  void *check_result_;
  float mp_scaling_;
  bool mp_enabled_ = false;

  StopCheck stop_check_;
  bool print_progress_ = true;
  bool print_progress_pending_nl_ = false;
  time_t print_progress_ts_ = 0;

  void finalize();

  std::shared_ptr<CudaTensor> resolveTensor_locked(std::shared_ptr<Tensor> t);

  std::shared_ptr<CudaTensor> resolveTensorGradient_locked(std::shared_ptr<Tensor> src);

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

  void progress(const char *what, long i, long batches, float mp_scaling);

  void progressDone();


};


class CudaOperation {
public:
  CudaOperation& operator=(CudaOperation const&) = delete;
  CudaOperation(CudaOperation const&) = delete;

  virtual ~CudaOperation() {}
  virtual const char *exec(CudaProgram &p, long batch) = 0;

  virtual std::vector<std::shared_ptr<CudaTensor>> getInputs() const = 0;
  virtual std::vector<std::shared_ptr<CudaTensor>> getOutputs() const = 0;
  virtual bool killOutput(std::shared_ptr<CudaTensorStorage> t) {
    return false;
  }

  virtual std::shared_ptr<CudaOperation> getSyncOp() {
    return nullptr;
  };

  void print(bool full = false) const;

  std::string name() const { return kind_; }

  virtual std::string info() const { return ""; };

  std::string str() const;

protected:

  CudaOperation(std::string kind) : kind_(kind) {}

  CudaOperation() = delete;

  std::string kind_;
};



#define CPPGLUE(a, b) a ## b
#define CPPJOIN(a, b) CPPGLUE(a, b)

void CudaRegisterOpFactory(const char *name,
                           const char *(*setup)(CudaProgram &p, const Node &n,
                                                bool training));

#define REGISTER_CUDA_OP(name, setup)                                   \
  static void __attribute__((constructor)) CPPJOIN(init, __LINE__)(void) { \
    CudaRegisterOpFactory(name, setup);                                 \
  }


typedef std::vector<std::shared_ptr<Node>> Nodes;

enum CudaTransformType {
  CUDA_TRANSFORM_ALL,
  CUDA_TRANSFORM_TRAINING,
  CUDA_TRANSFORM_INFERENCE,
};



void CudaRegisterTransform(CudaTransformType type,
                           Nodes (*op)(CudaProgram &p, const Nodes &nodes));

#define REGISTER_CUDA_TRANSFORM(prio, type, op)                         \
  static void __attribute__((constructor(1000 + prio)))                 \
  CPPJOIN(init, __LINE__)(void) {                                       \
    CudaRegisterTransform(type, op);                                    \
  }
}
