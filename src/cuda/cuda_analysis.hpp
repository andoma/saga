// -*-c++-*-

namespace saga {


struct CudaMemoryLayout {

  std::vector<std::pair<std::shared_ptr<CudaTensorStorage>, size_t>> table_;
  size_t size_;
};

std::vector<std::shared_ptr<CudaOperation>> reduceLiveranges(const std::vector<std::shared_ptr<CudaOperation>> &ops,
                                                             const std::vector<std::shared_ptr<CudaTensorStorage>> &exported);

std::unique_ptr<CudaMemoryLayout> memoryLayout(const std::vector<std::shared_ptr<CudaOperation>> &ops,
                                               const std::vector<std::shared_ptr<CudaTensorStorage>> &exported);

}
