#include <map>

#include <assert.h>
#include "saga.hpp"
#include "tensor.hpp"
#include "context.hpp"

#include "cuda_common.hpp"
#include "cuda_tensor.hpp"
#include "cuda_analysis.hpp"

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

static int __attribute__((unused)) bitchk(const uint32_t *bs, int v)
{
    if(bs[v >> 5] & (1 << (v & 31)))
        return 1;
    return 0;
}

static void
bitset_or(uint32_t *bs, const uint32_t *src, int words)
{
    for(int i = 0; i < words; i++) bs[i] |= src[i];
}

static void
bitset_andinv(uint32_t *bs, const uint32_t *src, int words)
{
    for(int i = 0; i < words; i++) bs[i] &= ~src[i];
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

struct Bestfit {  // Best fit allocator

    size_t m_total = 0;
    size_t m_peak = 0;
    std::multimap<size_t, size_t> m_sizes;
    std::map<size_t, size_t> m_chunks;

    size_t alloc(size_t reqsize)
    {
        size_t offset;
        auto it = m_sizes.upper_bound(reqsize);
        if(it == m_sizes.end()) {
            offset = m_total;
            m_total += reqsize;
            m_peak = std::max(m_total, m_peak);
        } else {
            size_t size = it->first;
            offset = it->second;

            m_chunks.erase(offset);

            m_sizes.erase(it);

            if(size > reqsize) {
                // Didn't use entire free chunk, reinsert remainder
                m_sizes.insert(
                    std::make_pair(size - reqsize, offset + reqsize));
                m_chunks.insert(
                    std::make_pair(offset + reqsize, size - reqsize));
            }
        }
        return offset;
    }

    void free(size_t position, size_t size)
    {
        auto next = m_chunks.find(position + size);
        if(next != m_chunks.end()) {
            // Merge next block
            eraseInSize(next->second, next->first);
            size += next->second;
            m_chunks.erase(next);
        }

        auto prev = m_chunks.lower_bound(position);
        if(prev != m_chunks.begin()) {
            // Merge prev block
            prev--;
            if(prev->first + prev->second == position) {
                eraseInSize(prev->second, prev->first);
                size += prev->second;
                position = prev->first;
                m_chunks.erase(prev);
            }
        }

        if(size + position == m_total) {
            m_total = position;
        } else {
            m_sizes.insert(std::make_pair(size, position));
            m_chunks.insert(std::make_pair(position, size));
        }
    }

    void eraseInSize(size_t size, size_t position)
    {
        auto range = m_sizes.equal_range(size);
        int found = 0;
        for(auto it = range.first; it != range.second;) {
            if(it->second == position) {
                it = m_sizes.erase(it);
                found++;
            } else {
                it++;
            }
        }
        assert(found == 1);
    }

    void dump()
    {
        printf("Bestfit heap peak:%zd current:%zd\n", m_peak, m_total);
        printf("chunks\n");
        for(auto it : m_chunks) {
            printf("\t%zd + %zd\n", it.first, it.second);
        }

        printf("sizes\n");
        for(auto it : m_sizes) {
            printf("\t%zd @ %zd\n", it.first, it.second);
        }
    }
};

struct Liveness {
    Liveness(Liveness const &) = delete;
    Liveness &operator=(Liveness const &) = delete;

    Liveness() = delete;

    Liveness(size_t words)
    {
        m_out = (uint32_t *)calloc(words, sizeof(uint32_t));
        m_in = (uint32_t *)calloc(words, sizeof(uint32_t));
        m_gen = (uint32_t *)calloc(words, sizeof(uint32_t));
        m_def = (uint32_t *)calloc(words, sizeof(uint32_t));
    }

    ~Liveness()
    {
        free(m_out);
        free(m_in);
        free(m_gen);
        free(m_def);
    }

    uint32_t *m_out;
    uint32_t *m_in;

    // Set of varibles used before any assignment (input tensors)
    uint32_t *m_gen;

    // Set of variables written to (output tensors)
    uint32_t *m_def;
};

struct LiveAnalysis {
    std::vector<std::shared_ptr<CudaOperation>> m_ops;

    int m_ln_id_base;
    int m_ln_words;
    int m_ln_size;
    std::vector<std::shared_ptr<Liveness>> m_lns;
    std::vector<size_t> m_memory_usage;
    std::vector<std::shared_ptr<CudaTensorStorage>> m_storage;

    uint32_t *m_exported_gen;

    LiveAnalysis(
        const std::vector<std::shared_ptr<CudaOperation>> &ops,
        const std::vector<std::shared_ptr<CudaTensorStorage>> &exported)
      : m_ops(ops)
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
            const int id = s->m_id;
            lowest_id = std::min(lowest_id, id);
            highest_id = std::max(highest_id, id);
        }

        m_ln_id_base = lowest_id;
        m_ln_size = highest_id - lowest_id + 1;
        m_ln_words = (m_ln_size + 31) / 32;

        m_lns.reserve(ops.size());

        m_memory_usage.resize(m_ln_size);
        m_storage.resize(m_ln_size);

        m_exported_gen = (uint32_t *)calloc(m_ln_words, sizeof(uint32_t));

        for(const auto &s : exported) {
            const int id = s->m_id - m_ln_id_base;
            bitset(m_exported_gen, id);
        }

        for(const auto &op : ops) {
            auto ln = std::make_shared<Liveness>(m_ln_words);

            for(const auto &t : op->getInputs()) {
                const int id = t->storage_id() - m_ln_id_base;
                bitset(ln->m_gen, id);
                m_memory_usage[id] = t->memoryUsage();
                m_storage[id] = t->m_storage;
            }

            for(const auto &t : op->getOutputs()) {
                const int id = t->storage_id() - m_ln_id_base;
                bitset(ln->m_def, id);
                m_memory_usage[id] = t->memoryUsage();
                m_storage[id] = t->m_storage;

                // This is a bit of a hack.
                // If writing to an offset in the tensor's storage we assume
                // it's the non-first part of a concat (or similar operation)
                if(t->m_offset)
                    bitset(ln->m_gen, id);
            }
            m_lns.push_back(ln);
        }
        update();
    }

    ~LiveAnalysis() { free(m_exported_gen); }

    void update()
    {
        uint32_t in_prim[m_ln_words];
        uint32_t out_buf[m_ln_words];

        if(m_ops.size() == 0)
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

            Liveness *succ = m_lns[0].get();
            memcpy(out_buf, succ->m_in, m_ln_words * sizeof(uint32_t));
            bitset_or(out_buf, m_exported_gen, m_ln_words);
            out_prim = out_buf;

            for(ssize_t i = m_ops.size() - 1; i >= 0; i--) {
                Liveness *cur = m_lns[i].get();

                for(int j = 0; j < m_ln_words; j++)
                    in_prim[j] = (out_prim[j] & ~cur->m_def[j]) | cur->m_gen[j];

                if(memcmp(cur->m_out, out_prim,
                          m_ln_words * sizeof(uint32_t)) ||
                   memcmp(cur->m_in, in_prim, m_ln_words * sizeof(uint32_t))) {
                    stable = 0;
                    memcpy(cur->m_out, out_prim, m_ln_words * sizeof(uint32_t));
                    memcpy(cur->m_in, in_prim, m_ln_words * sizeof(uint32_t));
                }

                out_prim = cur->m_in;
            }
        }
    }

    void eliminateDeadInstructions()
    {
        while(1) {
            bool did_something = false;
            for(ssize_t i = m_ops.size() - 1; i >= 0; i--) {
                bool kill = true;
                for(int j = 0; j < m_ln_words; j++) {
                    if(m_lns[i]->m_out[j] & m_lns[i]->m_def[j]) {
                        kill = false;
                        break;
                    }
                }
                if(kill) {
                    m_ops.erase(m_ops.begin() + i);
                    m_lns.erase(m_lns.begin() + i);
                    did_something = true;
                } else {
                    for(int j = 0; j < m_ln_words; j++) {
                        if((m_lns[i]->m_out[j] & m_lns[i]->m_def[j]) !=
                           m_lns[i]->m_def[j]) {
                            uint32_t out = m_lns[i]->m_out[j];
                            uint32_t def = m_lns[i]->m_def[j];
                            for(int k = 0; k < 32; k++) {
                                if(((1 << k) & def) && !((1 << k) & out)) {
                                    int id = j * 32 + k;
                                    auto s = m_storage[id];
                                    if(!m_ops[i]->killOutput(s)) {
                                        fprintf(
                                            stderr,
                                            "Unable to kill dead output T%d\n",
                                            id + m_ln_id_base);
                                        m_ops[i]->print(true);
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

    void print(bool liveranges = false)
    {
        for(size_t i = 0; i < m_ops.size(); i++) {
            if(liveranges && i == 0) {
                printf("Live:");
                for(int j = 0; j < m_ln_size; j++) {
                    if(bitchk(m_lns[i]->m_in, j)) {
                        printf(" T%d", j + m_ln_id_base);
                    }
                }
                printf("\n");
            }

            printf("%4zd: %s\n", i, m_ops[i]->str().c_str());
            if(liveranges) {
                printf(" Live:");
                for(int j = 0; j < m_ln_size; j++) {
                    if(bitchk(m_lns[i]->m_out, j)) {
                        printf(" T%d", j + m_ln_id_base);
                    }
                }
                printf("\n");
            }
        }
    }

    size_t peakMemoryUsage()
    {
        size_t peak = 0;

        for(size_t i = 0; i < m_ops.size(); i++) {
            size_t s = 0;
            for(int j = 0; j < m_ln_size; j++) {
                if(bitchk(m_lns[i]->m_in, j) || bitchk(m_lns[i]->m_out, j)) {
                    s += m_memory_usage[j];
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

        auto op = m_ops[from];
        auto ln = m_lns[from];

        m_ops.erase(m_ops.begin() + from);
        m_lns.erase(m_lns.begin() + from);

        if(from < to) {
            to--;
        }
        m_ops.insert(m_ops.begin() + to, op);
        m_lns.insert(m_lns.begin() + to, ln);
    }

    // Try to move operations further up in program order if
    // it saves memory usage
    void reduceMemoryPressure(void)
    {
        size_t memory_use = peakMemoryUsage();

        for(ssize_t i = 0; i < (ssize_t)m_ops.size(); i++) {
            const uint32_t *gen = m_lns[i]->m_gen;
            ssize_t j = -1;
            for(j = i - 1; j >= 0; j--) {
                if(bitset_insersects(gen, m_lns[j]->m_def, m_ln_words))
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

    std::unique_ptr<CudaMemoryLayout> memoryLayout()
    {
        Bestfit bf;
        const size_t align = 16;

        std::unordered_map<int, size_t> positions;
        std::unordered_set<int> inuse;

        for(size_t i = 0; i < m_ops.size(); i++) {
            for(int j = 0; j < m_ln_size; j++) {
                if(bitchk(m_lns[i]->m_out, j) && !bitchk(m_lns[i]->m_in, j)) {
                    const size_t size = (m_memory_usage[j] + align - 1) / align;

                    assert(inuse.find(j) == inuse.end());
                    inuse.insert(j);
                    positions[j] = align * bf.alloc(size);
                }
            }

            for(int j = 0; j < m_ln_size; j++) {
                if(bitchk(m_lns[i]->m_in, j) && !bitchk(m_lns[i]->m_out, j)) {
                    const size_t size = (m_memory_usage[j] + align - 1) / align;

                    assert(inuse.find(j) != inuse.end());
                    inuse.erase(j);
                    bf.free(positions[j] / align, size);
                }
            }
        }

        auto r = std::make_unique<CudaMemoryLayout>();

        r->table_.reserve(positions.size());
        for(const auto &p : positions) {
            r->table_.push_back(std::make_pair(m_storage[p.first], p.second));
        }
        r->size_ = bf.m_peak * align;
        return r;
    }
};

std::vector<std::shared_ptr<CudaOperation>>
reduceLiveranges(
    const std::vector<std::shared_ptr<CudaOperation>> &ops,
    const std::vector<std::shared_ptr<CudaTensorStorage>> &exported)
{
    LiveAnalysis la(ops, exported);
    la.eliminateDeadInstructions();
    la.reduceMemoryPressure();
    return la.m_ops;
}

std::unique_ptr<CudaMemoryLayout>
memoryLayout(const std::vector<std::shared_ptr<CudaOperation>> &ops,
             const std::vector<std::shared_ptr<CudaTensorStorage>> &exported)
{
    LiveAnalysis la(ops, exported);
    return la.memoryLayout();
}

}  // namespace saga
