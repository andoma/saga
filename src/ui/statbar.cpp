#include "saga.hpp"

#include <cmath>

namespace saga {

struct StatBar : public UI {
    void updateBatchInfo(enum Phase phase, int batch_size, int total_batches,
                         int epoch) override
    {
        m_phase = phase;
        m_batch_size = batch_size;
        m_total_batches = total_batches;
        m_current_epoch = epoch;
        if(!m_start)
            m_start = Now();
        refresh(m_start);
    }

    void updateProgress(int current_batch, int64_t total_samples) override
    {
        m_current_batch = current_batch;
        m_total_samples[m_phase] = total_samples;
        maybe_refresh();
    }

    void updateLoss(double loss) override
    {
        if(std::isfinite(m_loss)) {
            m_loss += (loss - m_loss) * 0.99f;
        } else {
            m_loss = loss;
        }
        maybe_refresh();
    }

    void updateMemUsage(size_t use, size_t total) override
    {
        m_mem_use = use;
        m_mem_total = total;
        maybe_refresh();
    }

    void updateMpScaling(double scaling) override
    {
        m_mp_scaling = scaling;
        maybe_refresh();
    }

    void maybe_refresh(void)
    {
        int64_t now = Now();
        if(now > m_last_update + 500000) {
            refresh(now);
        }
    }

    void refresh(int64_t now)
    {
        m_last_update = now;
        printf("\033[K");

        printf("%s | E: %-4d | B: %5d/%-5d N:%-4d | Mem %6zu/%-6zu",
               m_phase == TRAIN ? "Train" : "Infer", m_current_epoch,
               m_current_batch, m_total_batches, m_batch_size, m_mem_use >> 20,
               m_mem_total >> 20);

        if(m_total_samples[TRAIN]) {
            const int64_t total_samples = m_total_samples[TRAIN];
            printf(" | Samples/s: %6.2f",
                   total_samples * 1e6 / (double)(now - m_start));
        }

        if(std::isfinite(m_loss)) {
            printf(" | L: %8.4f", m_loss);
        }

        if(std::isfinite(m_mp_scaling)) {
            printf(" | MPS: %8.4f", m_mp_scaling);
        }

        printf("\r");
        fflush(stdout);
    }

    enum Phase m_phase { TRAIN };
    int m_batch_size{1};
    int m_total_batches{0};
    int m_current_batch{0};
    int m_current_epoch{0};
    int64_t m_total_samples[2] = {};
    double m_loss{NAN};
    size_t m_mem_use{0};
    size_t m_mem_total{0};
    double m_mp_scaling{NAN};

    int64_t m_last_update{0};
    int64_t m_start{0};
};

std::shared_ptr<UI> make_statbar()
{
    return std::make_shared<StatBar>();
}

}  // namespace saga
