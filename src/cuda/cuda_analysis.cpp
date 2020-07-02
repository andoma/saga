#include <map>

#include <assert.h>
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

// This should be made into a class I suppose
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



struct Bestfit { // Best fit allocator

  size_t total_ = 0;
  size_t peak_ = 0;
  std::multimap<size_t, size_t> sizes_;
  std::map<size_t, size_t> chunks_;

  size_t alloc(size_t reqsize)
  {
    size_t offset;
    auto it = sizes_.upper_bound(reqsize);
    if(it == sizes_.end()) {
      offset = total_;
      total_ += reqsize;
      peak_ = std::max(total_, peak_);
    } else {

      size_t size = it->first;
      offset = it->second;

      chunks_.erase(offset);

      sizes_.erase(it);

      if(size > reqsize) {
        // Didn't use entire free chunk, reinsert remainder
        sizes_.insert(std::make_pair(size - reqsize, offset + reqsize));
        chunks_.insert(std::make_pair(offset + reqsize, size - reqsize));
      }
    }
    return offset;
  }


  void free(size_t position, size_t size)
  {
    auto next = chunks_.find(position + size);
    if(next != chunks_.end()) {
      // Merge next block
      eraseInSize(next->second, next->first);
      size += next->second;
      chunks_.erase(next);
    }

    auto prev = chunks_.lower_bound(position);
    if(prev != chunks_.begin()) {
      // Merge prev block
      prev--;
      if(prev->first + prev->second == position) {
        eraseInSize(prev->second, prev->first);
        size += prev->second;
        position = prev->first;
        chunks_.erase(prev);
      }
    }

    if(size + position == total_) {
      total_ = position;
    } else {
      sizes_.insert(std::make_pair(size, position));
      chunks_.insert(std::make_pair(position, size));
    }
  }


  void eraseInSize(size_t size, size_t position)
  {
    auto range = sizes_.equal_range(size);
    int found = 0;
    for(auto it = range.first; it != range.second;) {
      if(it->second == position) {
        it = sizes_.erase(it);
        found++;
      } else {
        it++;
      }
    }
    assert(found == 1);
  }

  void dump()
  {
    printf("Bestfit heap peak:%zd current:%zd\n",
           peak_, total_);
    printf("chunks\n");
    for(auto it : chunks_) {
      printf("\t%zd + %zd\n", it.first, it.second);
    }

    printf("sizes\n");
    for(auto it : sizes_) {
      printf("\t%zd @ %zd\n", it.first, it.second);
    }
  }

};


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
  std::vector<std::shared_ptr<CudaTensorStorage>> storage_;

  uint32_t *exported_gen_;

  LiveAnalysis(const std::vector<std::shared_ptr<CudaOperation>> &ops,
               const std::vector<std::shared_ptr<CudaTensorStorage>> &exported)
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

    for(const auto &s : exported) {
      const int id = s->id_;
      lowest_id = std::min(lowest_id, id);
      highest_id = std::max(highest_id, id);
    }

    ln_id_base_ = lowest_id;
    ln_size_ = highest_id - lowest_id + 1;
    ln_words_ = (ln_size_ + 31) / 32;

    lns_.reserve(ops.size());

    memory_usage_.resize(ln_size_);
    storage_.resize(ln_size_);

    exported_gen_ = (uint32_t *)calloc(ln_words_, sizeof(uint32_t));

    for(const auto &s : exported) {
      const int id = s->id_ - ln_id_base_;
      bitset(exported_gen_, id);
    }

    for(const auto &op : ops) {
      auto ln = std::make_shared<Liveness>(ln_words_);

      for(const auto &t : op->getInputs()) {
        const int id = t->storage_id() - ln_id_base_;
        bitset(ln->gen_, id);
        memory_usage_[id] = t->memoryUsage();
        storage_[id] = t->storage_;
      }

      for(const auto &t : op->getOutputs()) {
        const int id = t->storage_id() - ln_id_base_;
        bitset(ln->def_, id);
        memory_usage_[id] = t->memoryUsage();
        storage_[id] = t->storage_;

        // This is a bit of a hack.
        // If writing to an offset in the tensor's storage we assume
        // it's the non-first part of a concat (or similar operation)
        if(t->offset_)
          bitset(ln->gen_, id);

      }
      lns_.push_back(ln);
    }
    update();
  }

  ~LiveAnalysis()
  {
    free(exported_gen_);
  }

  void update() {
    uint32_t in_prim[ln_words_];
    uint32_t out_buf[ln_words_];

    if(ops_.size() == 0)
      return;

    int stable = 0;
    while(!stable) {
      stable = 1;

      // Simple live analysis as there is no real control flow graph,
      // it's rather just an infinite loop.
      // We merge any tensors we've exported via resolveTensor() as
      // being read at end of the loop. This makes sure they're kept alive
      // at the end when user might inspect.

      const uint32_t *out_prim;

      Liveness *succ = lns_[0].get();
      memcpy(out_buf, succ->in_, ln_words_ * sizeof(uint32_t));
      bitset_or(out_buf, exported_gen_, ln_words_);
      out_prim = out_buf;

      for(ssize_t i = ops_.size() - 1; i >= 0; i--) {
        Liveness *cur  = lns_[i].get();

        for(int j = 0; j < ln_words_; j++)
          in_prim[j] = (out_prim[j] & ~cur->def_[j]) | cur->gen_[j];

        if(memcmp(cur->out_, out_prim, ln_words_ * sizeof(uint32_t)) ||
           memcmp(cur->in_,  in_prim,  ln_words_ * sizeof(uint32_t))) {

          stable = 0;
          memcpy(cur->out_, out_prim, ln_words_ * sizeof(uint32_t));
          memcpy(cur->in_,  in_prim,  ln_words_ * sizeof(uint32_t));
        }

        out_prim = cur->in_;
      }
    }
  }


  void eliminateDeadInstructions() {

    while(1) {
      bool did_something = false;
      for(ssize_t i = ops_.size() - 1; i >= 0; i--) {
        bool kill = true;
        for(int j = 0; j < ln_words_; j++) {
          if(lns_[i]->out_[j] & lns_[i]->def_[j]) {
            kill = false;
            break;
          }
        }
        if(kill) {
          ops_.erase(ops_.begin() + i);
          lns_.erase(lns_.begin() + i);
          did_something = true;
        } else {
          for(int j = 0; j < ln_words_; j++) {
            if((lns_[i]->out_[j] & lns_[i]->def_[j]) != lns_[i]->def_[j]) {
              uint32_t out = lns_[i]->out_[j];
              uint32_t def = lns_[i]->def_[j];
              for(int k = 0; k < 32; k++) {
                if(((1 << k) & def) && !((1 << k) & out)) {
                  int id = j * 32 + k;
                  auto s = storage_[id];
                  if(!ops_[i]->killOutput(s)) {
                    fprintf(stderr, "Warning: Unable to kill dead output\n");
                    exit(1);
                  }
                }
              }
            }
          }
        }
      }
      if(!did_something) {
        break;
      }
      update();
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


  std::unique_ptr<CudaMemoryLayout>  memoryLayout()
  {
    Bestfit bf;
    const size_t align = 16;

    std::unordered_map<int, size_t> positions;
    std::unordered_set<int> inuse;

    for(size_t i = 0; i <  ops_.size(); i++) {
      for(int j = 0; j < ln_size_; j++) {
        if(bitchk(lns_[i]->out_, j) && !bitchk(lns_[i]->in_, j)) {
          const size_t size = (memory_usage_[j] + align - 1) / align;

          assert(inuse.find(j) == inuse.end());
          inuse.insert(j);
          positions[j] = align * bf.alloc(size);
        }
      }

      for(int j = 0; j < ln_size_; j++) {
        if(bitchk(lns_[i]->in_, j) && !bitchk(lns_[i]->out_, j)) {
          const size_t size = (memory_usage_[j] + align - 1) / align;

          assert(inuse.find(j) != inuse.end());
          inuse.erase(j);
          bf.free(positions[j] / align, size);
        }
      }
    }

    auto r = std::make_unique<CudaMemoryLayout>();

    r->table_.reserve(positions.size());
    for(const auto &p : positions) {
      r->table_.push_back(std::make_pair(storage_[p.first], p.second));
    }
    r->size_ = bf.peak_ * align;
    return r;
  }
};


std::vector<std::shared_ptr<CudaOperation>>
reduceLiveranges(const std::vector<std::shared_ptr<CudaOperation>> &ops,
                 const std::vector<std::shared_ptr<CudaTensorStorage>> &exported)
{
  LiveAnalysis la(ops, exported);
  la.eliminateDeadInstructions();
  la.reduceMemoryPressure();
  return la.ops_;
}



std::unique_ptr<CudaMemoryLayout>
memoryLayout(const std::vector<std::shared_ptr<CudaOperation>> &ops,
             const std::vector<std::shared_ptr<CudaTensorStorage>> &exported)
{
  LiveAnalysis la(ops, exported);
  return la.memoryLayout();
}

}
