// -*-c++-*-

#pragma once

#include <memory>
#include <mutex>

#include <cudnn.h>
#include <cublas_v2.h>


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
    int cuda_status__ = (expression);                                   \
    if(cuda_status__) {                                                 \
      fprintf(stderr, "CUDA error at %s:%d in %s\n",                    \
              __FILE__, __LINE__, __FUNCTION__);                        \
    }                                                                   \
  }


namespace saga {

class CudaTensor;
class CudaOperation;
class CudaTensorStorage;
class CudaTensorStorageDoubleBuffered;



class CudaContext : public Context,
                    public std::enable_shared_from_this<CudaContext> {
public:
  CudaContext()
    : cudnn_(NULL)
    , cublas_(NULL)
    , tensor_storage_id_gen_(0)
  {}

  ~CudaContext()
  {}

  int init();

  std::shared_ptr<Program> createProgram(const Graph &g,
                                         const ProgramConfig &pc,
                                         const BatchTensorAccessors &accessors);

  void print() override;

  cudaStream_t stream_;
  cudnnHandle_t cudnn_;
  cublasHandle_t cublas_;

  int deviceId_;
  std::mutex mutex_;

  int tensor_storage_id_gen_;
};


struct CudaBatchAccessOp {
  std::shared_ptr<CudaTensor> tensor_;
  std::shared_ptr<CudaTensorStorageDoubleBuffered> storage_;
  BatchTensorAccessFn fn_;
  bool prefetch_ = false;
};

typedef std::vector<CudaBatchAccessOp> CudaBatchAccessOps;



class CudaProgram : public Program {
public:

  CudaProgram(std::shared_ptr<CudaContext> ctx,
               TensorLayout tensor_layout,
               int batch_size,
               float learning_rate)
    : ctx_(ctx)
    , tensor_layout_(tensor_layout)
    , batch_size_(batch_size)
    , learning_rate_(learning_rate)
    , debug_(false)
    , workspace_(NULL)
    , workspace_size_(0)
    , workspace_requested_(0)
    , mp_scaling_(batch_size)
  {
    chkCuda(cudaMallocManaged(&check_result_, sizeof(int), cudaMemAttachGlobal));
    chkCuda(cudaMemset(check_result_, 0, sizeof(int)));
  }

  ~CudaProgram()
  {
    chkCuda(cudaFree(workspace_));
    chkCuda(cudaFree(check_result_));
  }


  std::shared_ptr<Tensor> resolveTensor(std::shared_ptr<Tensor> t) override;
  void infer(long batches) override;
  void train(long batches) override;
  void print() const override;
  void debug(bool) override;

  void requetstWorkspace(size_t size) {
    workspace_requested_ = std::max(workspace_requested_, size);
  }

  void allocWorkspace() {
    if(workspace_requested_ <= workspace_size_)
      return;

    chkCuda(cudaFree(workspace_));

    workspace_size_ = workspace_requested_;
    chkCuda(cudaMalloc(&workspace_, workspace_size_));
  }

  const std::shared_ptr<CudaContext> ctx_;
  const TensorLayout tensor_layout_;
  const int batch_size_;
  const float learning_rate_;
  bool debug_;

  std::unordered_map<std::shared_ptr<Tensor>,
                     std::shared_ptr<CudaTensor>> tensors_;

  std::unordered_map<std::shared_ptr<Tensor>,
                     std::shared_ptr<CudaOperation>> load_operations_;

  std::vector<std::shared_ptr<CudaOperation>> infer_operations_;
  std::vector<std::shared_ptr<CudaOperation>> train_operations_;

  std::vector<std::shared_ptr<CudaOperation>> fwd_operations_;
  std::vector<std::shared_ptr<CudaOperation>> bwd_operations_;
  std::vector<std::shared_ptr<CudaOperation>> upd_operations_;

  CudaBatchAccessOps infer_pre_;
  CudaBatchAccessOps infer_post_;
  CudaBatchAccessOps train_pre_;
  CudaBatchAccessOps train_post_;

  std::vector<std::shared_ptr<CudaTensorStorageDoubleBuffered>> flips_;

  void *workspace_;
  size_t workspace_size_;
  size_t workspace_requested_;

  void *check_result_;
  float mp_scaling_;

  std::shared_ptr<CudaTensor> resolveTensor_locked(std::shared_ptr<Tensor> t);

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

  void issueOps(const CudaBatchAccessOps ops, long batch);

  void flipDoubleBufferedTensors();

};


class CudaOperation {
public:
  CudaOperation& operator=(CudaOperation const&) = delete;
  CudaOperation(CudaOperation const&) = delete;

  virtual ~CudaOperation() {}
  virtual void exec(CudaProgram &p, long batch) = 0;

  virtual std::vector<std::shared_ptr<CudaTensor>> getInputs() const = 0;
  virtual std::vector<std::shared_ptr<CudaTensor>> getOutputs() const = 0;

  virtual std::shared_ptr<CudaOperation> getSyncOp() {
    return nullptr;
  };

  void print(bool full = false) const;

  std::string name() const { return kind_; }

  std::string str() const;

protected:

  CudaOperation(std::string kind) : kind_(kind) {}

  CudaOperation() = delete;

private:
  const std::string kind_;
};



#define CPPGLUE(a, b) a ## b
#define CPPJOIN(a, b) CPPGLUE(a, b)

void CudaRegisterOpFactory(const char *name,
                           void (*setup)(CudaProgram &p, const Node &n,
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
