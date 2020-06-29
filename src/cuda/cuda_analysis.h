// -*-c++-*-

namespace saga {


struct CudaMemoryLayout {

  std::vector<std::pair<std::shared_ptr<CudaTensorStorage>, size_t>> table_;
  size_t size_;
};

std::vector<std::shared_ptr<CudaOperation>> reduceLiveranges(std::vector<std::shared_ptr<CudaOperation>> &ops);

std::unique_ptr<CudaMemoryLayout> memoryLayout(std::vector<std::shared_ptr<CudaOperation>> &ops);

}
