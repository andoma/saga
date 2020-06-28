#include "saga.h"
#include "tensor.h"
#include "context.h"

#include "cuda_common.h"
#include "cuda_tensor.h"
#include "cuda_analysis.h"

namespace saga {


class CudaTensor;
class CudaOperation;
class CudaTensorStorage;


static void
bitset(uint32_t *bs, int v)
{
  bs[v >> 5] |= 1 << (v & 31);
}

static void
bitclr(uint32_t *bs, int v)
{
  bs[v >> 5] &= ~(1 << (v & 31));
}


static int __attribute__((unused))
bitchk(const uint32_t *bs, int v)
{
  if(bs[v >> 5] & (1 << (v & 31)))
    return 1;
  return 0;
}

static void
bitset_or(uint32_t *bs, const uint32_t *src, int words)
{
  for(int i = 0; i < words; i++)
    bs[i] |= src[i];
}

static void
bitset_andinv(uint32_t *bs, const uint32_t *src, int words)
{
  for(int i = 0; i < words; i++)
    bs[i] &= ~src[i];
}

static void
bitset_print(const uint32_t *bs, int words)
{
  for(int i = 0; i < words; i++) {
    printf("0x%08x ", bs[i]);
  }
  printf("\n");
}

static bool
bitset_insersects(const uint32_t *a, const uint32_t *b, int words)
{
  for(int i = 0; i < words; i++) {
    if(a[i] & b[i])
      return true;
  }
  return false;
}




struct Liveness {

  Liveness(Liveness const&) = delete;
  Liveness& operator=(Liveness const&) = delete;

  Liveness() = delete;

  Liveness(size_t words)
  {
    out_ = (uint32_t *)calloc(words, sizeof(uint32_t));
    in_  = (uint32_t *)calloc(words, sizeof(uint32_t));
    gen_ = (uint32_t *)calloc(words, sizeof(uint32_t));
    def_ = (uint32_t *)calloc(words, sizeof(uint32_t));
  }

  ~Liveness()
  {
    free(out_);
    free(in_);
    free(gen_);
    free(def_);
  }

  uint32_t *out_;
  uint32_t *in_;

  // Set of varibles used before any assignment (input tensors)
  uint32_t *gen_;

  // Set of variables written to (output tensors)
  uint32_t *def_;
};


struct LiveAnalysis {

  std::vector<std::shared_ptr<CudaOperation>> ops_;

  int ln_id_base_;
  int ln_words_;
  int ln_size_;
  std::vector<std::shared_ptr<Liveness>> lns_;
  std::vector<size_t> memory_usage_;

  LiveAnalysis(const std::vector<std::shared_ptr<CudaOperation>> &ops)
    : ops_(ops)
  {
    int lowest_id = INT32_MAX;
    int highest_id = INT32_MIN;

    for(const auto &op : ops) {
      for(const auto &t : op->getInputs()) {
        const int id = t->storage_id();
        lowest_id = std::min(lowest_id, id);
        highest_id = std::max(highest_id, id);
      }

      for(const auto &t : op->getOutputs()) {
        const int id = t->storage_id();
        lowest_id = std::min(lowest_id, id);
        highest_id = std::max(highest_id, id);
      }
    }

    ln_id_base_ = lowest_id;
    ln_size_ = highest_id - lowest_id + 1;
    ln_words_ = (ln_size_ + 31) / 32;

    lns_.reserve(ops.size());

    memory_usage_.resize(ln_size_);

    for(const auto &op : ops) {
      auto ln = std::make_shared<Liveness>(ln_words_);

      for(const auto &t : op->getInputs()) {
        const int id = t->storage_id() - ln_id_base_;
        bitset(ln->gen_, id);
        memory_usage_[id] = t->memoryUsage();
      }

      for(const auto &t : op->getOutputs()) {
        const int id = t->storage_id() - ln_id_base_;
        bitset(ln->def_, id);
        memory_usage_[id] = t->memoryUsage();
        if(t->offset_)
          bitset(ln->gen_, id);

      }
      lns_.push_back(ln);
    }
    update();
  }


  void update() {
    uint32_t in_prim[ln_words_];

    int stable = 0;
    while(!stable) {
      stable = 1;

      Liveness *succ = lns_[0].get();
      for(ssize_t i = ops_.size() - 1; i >= 0; i--) {
        Liveness *cur  = lns_[i].get();

        const uint32_t *out_prim = succ->in_;

        for(int i = 0; i < ln_words_; i++)
          in_prim[i] = (out_prim[i] & ~cur->def_[i]) | cur->gen_[i];

        if(memcmp(cur->out_, out_prim, ln_words_ * sizeof(uint32_t)) ||
           memcmp(cur->in_,  in_prim,  ln_words_ * sizeof(uint32_t))) {

          stable = 0;
          memcpy(cur->out_, out_prim, ln_words_ * sizeof(uint32_t));
          memcpy(cur->in_,  in_prim,  ln_words_ * sizeof(uint32_t));
        }

        succ = cur;
      }
    }
  }

  void print(bool liveranges = false) {

    for(size_t i = 0; i <  ops_.size(); i++) {
      if(liveranges && i == 0) {
        printf("Live:");
        for(int j = 0; j < ln_size_; j++) {
          if(bitchk(lns_[i]->in_, j)) {
            printf(" T%d", j + ln_id_base_);
          }
        }
        printf("\n");
      }

      printf("%4zd: %s\n", i, ops_[i]->str().c_str());
      if(liveranges) {

        printf(" Live:");
        for(int j = 0; j < ln_size_; j++) {
          if(bitchk(lns_[i]->out_, j)) {
            if(j + ln_id_base_ == 15)
               printf(" T%d", j + ln_id_base_);
          }
        }
        printf("\n");
      }
    }
  }

  size_t peakMemoryUsage() {

    size_t peak = 0;

    for(size_t i = 0; i <  ops_.size(); i++) {

      size_t s = 0;
      for(int j = 0; j < ln_size_; j++) {
        if(bitchk(lns_[i]->in_, j) || bitchk(lns_[i]->out_, j)) {
          s += memory_usage_[j];
        }
      }
      if(s > peak) {
        peak = s;
      }
    }
    return peak;
  }

  void moveOp(size_t from, size_t to)
  {
    if(from == to)
      return;

    auto op = ops_[from];
    auto ln = lns_[from];

    ops_.erase(ops_.begin() + from);
    lns_.erase(lns_.begin() + from);

    if(from < to) {
      to--;
    }
    ops_.insert(ops_.begin() + to, op);
    lns_.insert(lns_.begin() + to, ln);
  }

  // Try to move operations further up in program order if
  // it saves memory usage
  void reduceMemoryPressure(void)
  {
    size_t memory_use = peakMemoryUsage();

    for(ssize_t i = 0; i < (ssize_t)ops_.size(); i++) {
      const uint32_t *gen = lns_[i]->gen_;
      ssize_t j = -1;
      for(j = i - 1; j >= 0; j--) {
        if(bitset_insersects(gen, lns_[j]->def_, ln_words_))
          break;
      }

      if(j == -1)
        continue;

      j++;
      if(j == i)
        continue;

      moveOp(i, j);
      update();
      size_t new_memory_use = peakMemoryUsage();
      if(new_memory_use >= memory_use) {
        moveOp(j, i);
        continue;
      }
      memory_use = new_memory_use;
    }
  }
};


std::vector<std::shared_ptr<CudaOperation>>
reduceLiveranges(std::vector<std::shared_ptr<CudaOperation>> &ops)
{
  LiveAnalysis la(ops);
  la.reduceMemoryPressure();
  return la.ops_;
}

}
